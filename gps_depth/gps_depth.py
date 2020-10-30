import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul
import math

import time


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)  # 乘上b b是全一向量 相当于每行求和

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class GPSDepthAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True,thickness=1, max_dis=3, k=100, attention_size = 16, GPU = False, need_norm = True):
        super(GPSDepthAttentionLayer, self).__init__()
        self.GPU = GPU  # 是否使用GPU
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.attention_hid = attention_size  # 决定了attention的Wk Wq的shape
        self.la_simple = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_normal_(self.la_simple.data, gain=1.414)
        self.ra_simple = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_normal_(self.ra_simple.data, gain=1.414)
        self.Bla_simple = nn.Parameter(torch.zeros(size=(1, )))
        self.Bra_simple = nn.Parameter(torch.zeros(size=(1, )))

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
        self.B = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_normal_(self.B.data, gain=1.414)

        self.W_2mini = nn.Parameter(torch.zeros(size=(out_features, self.attention_hid)))
        nn.init.xavier_normal_(self.W_2mini.data, gain=1.414)
        self.B_2mini = nn.Parameter(torch.zeros(size=(1, self.attention_hid)))
        nn.init.xavier_normal_(self.B_2mini.data, gain=1.414)

        self.dropout = dropout
        # tmp
        self.attention_dropout = 0.05
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.k = k
        # self.linear = nn.Linear(in_features, out_features, bias=True)
        self.thickness = thickness
        self.need_norm = need_norm
        if need_norm:
            self.bn = torch.nn.BatchNorm1d(self.out_features)
            self.bn.reset_parameters()

        self.linear_factor1 = nn.Linear(self.attention_hid*3, self.attention_hid)
        nn.init.xavier_normal_(self.linear_factor1.weight, gain=1.414)
        nn.init.constant_(self.linear_factor1.bias, 0)

        self.linear_factor2 = nn.Linear(self.attention_hid, 1)
        nn.init.xavier_normal_(self.linear_factor2.weight, gain=1.414)
        nn.init.constant_(self.linear_factor2.bias, 0)
        self.special_spmm = SpecialSpmm()


    def forward(self, input, adj, aggr_factor, edges):
        # --------------------------------------------------------------------------------
        # adj: sparse tensor
        # aggr_factor：(N, ) tensor. float
        # --------------------------------------------------------------------------------
        N = input.size()[0]         # 此时N为实际的N+1(多了一个special_id)
        # 对h做线性变换
        new_h = torch.mm(input, self.W) + self.B

        # 先算出simple_attention的la和ra，为后续做准备
        simple_attention_la = torch.mm(new_h, self.la_simple).view(-1) + self.Bla_simple
        simple_attention_ra = torch.mm(new_h, self.ra_simple).view(-1) + self.Bra_simple
        F.dropout(simple_attention_la, self.attention_dropout, training=self.training)
        F.dropout(simple_attention_ra, self.attention_dropout, training=self.training)

        a_src = torch.index_select(simple_attention_la, 0, edges[0])
        a_dst = torch.index_select(simple_attention_ra, 0, edges[1])
        a_edge = (a_src + a_dst).div(math.sqrt(self.out_features))
        a_edge = torch.exp(- self.leakyrelu(a_edge))
        # test!
        a_sparse = SparseTensor.from_edge_index(edge_index=edges, edge_attr=a_edge, sparse_sizes=torch.Size([N, N])).cuda()
        a_sparse_sum_rowwise = matmul(a_sparse, torch.ones(N, 1).cuda(), reduce='sum')

        final_h = aggr_factor * (torch.div(matmul(a_sparse, new_h), a_sparse_sum_rowwise)) + (1-aggr_factor) * new_h

        if self.need_norm:
            # batch normalization
            final_h = self.bn(final_h)
        # test
        if self.thickness != 3:
            # 若不是最后一层
            final_h = F.relu(final_h, inplace=True)
            final_h = F.dropout(final_h, self.dropout, training=self.training)

        new_h_mini = torch.mm(final_h, self.W_2mini) + self.B_2mini

        # get aggre_factor for next layer
        h_src = torch.index_select(new_h_mini, 0, edges[0])
        h_dst = torch.index_select(new_h_mini, 0, edges[1])
        h_diff = torch.abs(h_dst - h_src)
        factor_cal = torch.cat((h_src, h_dst, h_diff), 1)
        factor_cal = self.linear_factor1(factor_cal)
        # factor_cal = factor_cal.div(self.out_features)
        factor_cal = F.tanh(factor_cal)
        factor_cal = self.linear_factor2(factor_cal)
        # factor_cal = factor_cal.div(self.out_features)
        factor_cal = F.sigmoid(factor_cal).squeeze()

        factor_cal_sparse = SparseTensor.from_edge_index(edge_index=edges, edge_attr=factor_cal, sparse_sizes=torch.Size([N, N])).cuda()
        factor_res_1hop = matmul(factor_cal_sparse, torch.ones(N, 1).cuda(), reduce='mean')
        factor_res_2hop = matmul(factor_cal_sparse, factor_res_1hop, reduce='mean')

        return final_h, factor_res_2hop

    def __repr__(self):
        return (
                self.__class__.__name__
                + " ("
                + str(self.in_features)
                + " -> "
                + str(self.out_features)
                + ")"
        )


class GPSDepth(nn.Module):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--hidden-size", type=int, default=256)
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument("--alpha", type=float, default=0.2)
        parser.add_argument("--nheads", type=int, default=1)

        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.hidden_size,
            args.num_classes,
            args.dropout,
            args.alpha,
            args.nheads,
            args.layers
        )

    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, layers=3,):
        """Sparse version of GAT."""
        super(GPSDepth, self).__init__()
        self.GPU = True
        self.dropout = dropout
        self.heads = nheads     # head: 感受野不同，最后一层cat起来
        self.layers = layers    # 层数
        self.attentions = []
        self.subheads = 1       # subhead: 同一感受野内部，attention的W矩阵不同
        self.nclass = nclass
        self.attention_size = 16

        # 第一层： in: nfeat; out: nhid，即参数中的hidden-size
        self.attentions.append([
            [GPSDepthAttentionLayer(
                nfeat, nhid, attention_size=self.attention_size,  dropout=dropout, alpha=alpha, concat=True, thickness=1, GPU=self.GPU, need_norm=True
            ) for __ in range(self.subheads)]
            for _ in range(nheads)
        ])

        # 第2-(n-1)层 in和out均为: nhid * 每种感受野内部的subhead数
        for i in range(layers - 2):
            self.attentions.append([
                [GPSDepthAttentionLayer(
                    nhid, nhid, attention_size=self.attention_size, dropout=dropout, alpha=alpha, concat=True, thickness=i + 2, GPU=self.GPU, need_norm=True
                ) for __ in range(self.subheads)]
                for _ in range(nheads)
            ])

        # 第n层： in: nhid * 每种感受野内部的subhead数; out: nclass,即类别数
        self.attentions.append([
            [GPSDepthAttentionLayer(
                nhid, nclass, attention_size=self.attention_size, dropout=dropout, alpha=alpha, concat=True, thickness=layers, GPU=self.GPU, need_norm=False
            ) for __ in range(self.subheads)]
            for _ in range(nheads)
        ])

        for i, attention in enumerate(self.attentions):
            for j, att in enumerate(attention):
                for t, at in enumerate(att):
                    self.add_module("attention_{}_{}_{}".format(i, j, t), at)

    def forward(self, x, adj, edges):
        # adj: sparse tensor
        y = [x.clone() for _ in range(self.heads)]
        z = [x.clone() for _ in range(self.heads)]
        N = x.size()[0]
        aggr_factors = [torch.ones(N, 1).cuda() for _ in range(self.heads)]

        for layer in range(self.layers):
            # head: 感受野不同，最后一层cat起来
            for number_attention in range(self.heads):
                # subhead: 共享感受野，但attention的W矩阵不同
                if layer == 3:
                    print("layer", layer, aggr_factors[0].squeeze().sum() / N)
                for number_subattention in range(self.subheads):
                    ytmp, aggr_factors_tmp = \
                        self.attentions[layer][number_attention][number_subattention](y[number_attention],
                                                                                      adj, aggr_factors[number_attention], edges)
                    aggr_factors[number_attention] = aggr_factors[number_attention] * aggr_factors_tmp
                    if number_subattention == 0:
                        z[number_attention] = ytmp
                    else:
                        z[number_attention] = z[number_attention] + ytmp
                        # z[number_attention] = torch.cat((z[number_attention], ytmp), 1)
                y[number_attention] = z[number_attention]


        # print("attention", receptive_field[0][:, 119, :])
        # 把不同head的输出stack起来，并求和
        x = torch.stack(y)
        x = x.sum(0)
        return F.log_softmax(x, dim=1)

    def loss(self, datax, datay, train_mask, adj_sparse, edges):

        return F.nll_loss(
            self.forward(datax, adj_sparse, edges)[train_mask],
            datay[train_mask],
        )

    def predict(self, datax, adj_sparse, edges):
        return self.forward(datax, adj_sparse, edges)
