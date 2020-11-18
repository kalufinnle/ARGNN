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

        self.Linear_2mini = nn.Linear(in_features, self.attention_hid)
        # self.W_2mini = nn.Parameter(torch.zeros(size=(in_features, self.attention_hid)))
        nn.init.xavier_normal_(self.Linear_2mini.weight, gain=1.414)
        # self.B_2mini = nn.Parameter(torch.zeros(size=(1, self.attention_hid)))
        nn.init.constant_(self.Linear_2mini.bias, 0)

        self.attention_bias = nn.Parameter(torch.zeros(size=(1, attention_size)))
        nn.init.constant_(self.attention_bias, 0)

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


    def forward(self, input:torch.Tensor, adj, edge_factor, edges, adj_sparse_sum_rowwise, degree, iftrain):
        # --------------------------------------------------------------------------------
        # adj: sparse tensor
        # aggr_factor：(N, ) tensor. float
        # --------------------------------------------------------------------------------
        N = input.size()[0]

        new_h_mini = self.Linear_2mini(input)
        h_src = torch.index_select(new_h_mini, 0, edges[0])
        h_dst = torch.index_select(new_h_mini, 0, edges[1])
        h_diff = torch.abs(h_dst - h_src)
        factor_cal = torch.cat((h_src, h_dst, h_diff), 1)
        factor_cal = self.linear_factor1(factor_cal)
        factor_cal = self.leakyrelu(factor_cal)
        factor_cal = self.linear_factor2(factor_cal)
        factor_cal_0 = F.sigmoid(factor_cal).squeeze()

        factor_cal_sparse_0 = SparseTensor.from_edge_index(edge_index=edges, edge_attr=factor_cal_0, sparse_sizes=torch.Size([N, N])).cuda()
        factor_res_1hop = matmul(factor_cal_sparse_0, torch.ones(N, 1).cuda(), reduce='mean')
        # factor_res_2hop = matmul(factor_cal_sparse_0, factor_res_1hop, reduce='mean')
        factor_src = torch.index_select(factor_res_1hop, 0, edges[0]).squeeze()
        factor_dst = torch.index_select(factor_res_1hop, 0, edges[1]).squeeze()

        h_src = h_src + self.attention_bias
        h_dst = h_dst + self.attention_bias
        factor_cal = torch.cat((h_src, h_dst, h_diff), 1)
        factor_cal = self.linear_factor1(factor_cal)
        factor_cal = self.leakyrelu(factor_cal)
        factor_cal = self.linear_factor2(factor_cal)
        factor_cal_1 = F.sigmoid(factor_cal).squeeze()

        # edge_factor = edge_factor * factor_src * factor_dst * factor_cal_1
        edge_factor = factor_src * factor_dst * factor_cal_1
        # edge_factor = factor_cal_1

        edge_factor_sparse = SparseTensor.from_edge_index(edge_index=edges, edge_attr=edge_factor, sparse_sizes=torch.Size([N, N])).cuda()
        edge_factor_sparse_sum_rowwise = matmul(edge_factor_sparse, torch.ones(N, 1).cuda(), reduce='sum')
        aggr_factor = torch.div(edge_factor_sparse_sum_rowwise, adj_sparse_sum_rowwise)

        degree = degree.unsqueeze(1)
        final_h = matmul(edge_factor_sparse, input*degree)*degree  + (1-aggr_factor) * input

        return final_h, edge_factor

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

    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, layers,):
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
        self.alpha = alpha

        self.linear_before = nn.Linear(nfeat, nhid)
        nn.init.xavier_normal_(self.linear_before.weight, gain=1.414)
        nn.init.constant_(self.linear_before.bias, 0)

        # 第一层： in: nfeat; out: nhid，即参数中的hidden-size
        self.attentions.append([
            [GPSDepthAttentionLayer(
                nhid, nhid, attention_size=self.attention_size,  dropout=dropout, alpha=alpha, concat=True, thickness=1, GPU=self.GPU, need_norm=True
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

        self.linear_after1 = nn.Linear(nhid, nhid)
        self.linear_after2 = nn.Linear(nhid, nclass)
        nn.init.xavier_normal_(self.linear_after1.weight, gain=1.414)
        nn.init.constant_(self.linear_after1.bias, 0)
        nn.init.xavier_normal_(self.linear_after2.weight, gain=1.414)
        nn.init.constant_(self.linear_after2.bias, 0)


    def forward(self, x, adj, edges, degree, iftrain):
        # adj: sparse tensor
        x = self.linear_before(x)
        # x = F.leaky_relu(x, self.alpha)
        N = x.size()[0]
        N_edge = edges.size()[1]

        adj_sparse_sum_rowwise = matmul(adj, torch.ones(N, 1).cuda(), reduce='sum')

        y = [x.clone() for _ in range(self.heads)]
        z = [x.clone() for _ in range(self.heads)]
        edge_factors = [torch.ones(N_edge).cuda() for _ in range(self.heads)]


        for layer in range(self.layers):
            # head: 感受野不同，最后一层cat起来
            for number_attention in range(self.heads):
                # subhead: 共享感受野，但attention的W矩阵不同
                # if layer != 0 and iftrain:
                    # print("layer", layer, aggr_factors[0][0:10])
                    # print("layer", layer, aggr_factors[0].squeeze().sum() / N)
                for number_subattention in range(self.subheads):
                    ytmp, edge_factors[number_attention] = \
                        self.attentions[layer][number_attention][number_subattention](y[number_attention],
                                                                                      adj, edge_factors[number_attention], edges, adj_sparse_sum_rowwise, degree, iftrain)
                    if iftrain:
                        print(edge_factors[0][:10])
                    if number_subattention == 0:
                        z[number_attention] = ytmp
                    else:
                        z[number_attention] = z[number_attention] + ytmp
                y[number_attention] = z[number_attention]


        # print("attention", receptive_field[0][:, 119, :])
        # 把不同head的输出stack起来，并求和

        for number_attention in range(self.heads):
            y[number_attention] = self.linear_after1(y[number_attention])
            y[number_attention] = F.leaky_relu(y[number_attention], self.alpha)
            y[number_attention] = self.linear_after2(y[number_attention])


        x = torch.stack(y)
        x = x.sum(0)
        return F.log_softmax(x, dim=1)

    def loss(self, datax, datay, train_mask, adj_sparse, edges, degree):

        return F.nll_loss(
            self.forward(datax, adj_sparse, edges, degree, True)[train_mask],
            datay[train_mask],
        )

    def predict(self, datax, adj_sparse, edges, degree):
        return self.forward(datax, adj_sparse, edges, degree, False)
