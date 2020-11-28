import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv

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

    def __init__(self, in_features, out_features, dropout, alpha, N, concat=True,thickness=1, max_dis=3, k=100, attention_size = 16, need_norm = True):
        super(GPSDepthAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conv = GCNConv(in_features, out_features, cached=True)
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

        self.attention_bias = nn.Parameter(torch.zeros(size=(1, attention_size+1)))
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

        self.linear_factor1 = nn.Linear((self.attention_hid+1)*3, self.attention_hid)
        nn.init.xavier_normal_(self.linear_factor1.weight, gain=1.414)
        nn.init.constant_(self.linear_factor1.bias, 0)

        self.linear_factor2 = nn.Linear(self.attention_hid, 1)
        nn.init.xavier_normal_(self.linear_factor2.weight, gain=1.414)
        nn.init.constant_(self.linear_factor2.bias, 0)
        self.special_spmm = SpecialSpmm()


    def forward(self, input:torch.Tensor, adj, aggr_factor, edges, adj_sparse_sum_rowwise, degree, iftrain, LP_W):
        # --------------------------------------------------------------------------------
        # adj: sparse tensor
        # aggr_factor：(N, ) tensor. float
        # --------------------------------------------------------------------------------
        N = input.size()[0]
        adjust_input = LP_W * input
        final_h = self.conv(adjust_input, adj)
        # final_h = self.conv(input, adj)
        final_h = aggr_factor * final_h + (1-aggr_factor) * input

        new_h_mini = self.Linear_2mini(final_h)
        #杂乱程度
        adj_sparse_sum_rowwise = matmul(adj, torch.ones(N, 1).cuda(), reduce='sum')
        avg_nei_feature = torch.div(matmul(adj, new_h_mini), adj_sparse_sum_rowwise)
        left_feature = torch.index_select(avg_nei_feature, 0, edges[0])
        right_feature = torch.index_select(new_h_mini, 0, edges[1])
        distance = F.pairwise_distance(left_feature, right_feature)
        distance_adj = SparseTensor.from_edge_index(edge_index=edges, edge_attr=distance, sparse_sizes=torch.Size([N, N])).cuda()
        distance_2 = matmul(distance_adj, torch.ones(N, 1).cuda(), reduce='mean')
        new_h_mini = torch.cat((new_h_mini, distance_2), dim=1)

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
        factor_cal_0 = F.sigmoid(factor_cal).squeeze()
        # if iftrain:
        #     print("factor_cal_0")
        #     print(factor_cal_0[0:30])

        new_h_mini = self.Linear_2mini(input)
        adj_sparse_sum_rowwise = matmul(adj, torch.ones(N, 1).cuda(), reduce='sum')
        avg_nei_feature = torch.div(matmul(adj, new_h_mini), adj_sparse_sum_rowwise)
        left_feature = torch.index_select(avg_nei_feature, 0, edges[0])
        right_feature = torch.index_select(new_h_mini, 0, edges[1])
        distance = F.pairwise_distance(left_feature, right_feature)
        distance_adj = SparseTensor.from_edge_index(edge_index=edges, edge_attr=distance, sparse_sizes=torch.Size([N, N])).cuda()
        distance_2 = matmul(distance_adj, torch.ones(N, 1).cuda(), reduce='mean')
        new_h_mini = torch.cat((new_h_mini, distance_2), dim=1)

        h_src = torch.index_select(new_h_mini, 0, edges[0])
        h_dst = torch.index_select(new_h_mini, 0, edges[1])
        h_src = h_src + self.attention_bias
        h_dst = h_dst + self.attention_bias
        # print("bias", self.attention_bias)
        factor_cal = torch.cat((h_src, h_dst, h_diff), 1)
        factor_cal = self.linear_factor1(factor_cal)
        # factor_cal = factor_cal.div(self.out_features)
        factor_cal = F.tanh(factor_cal)
        factor_cal = self.linear_factor2(factor_cal)
        # factor_cal = factor_cal.div(self.out_features)
        factor_cal_1 = F.sigmoid(factor_cal).squeeze()
        # if iftrain:
        #     print("factor_cal_1")
        #     print(factor_cal_1[0:30])

        factor_cal_sparse_0 = SparseTensor.from_edge_index(edge_index=edges, edge_attr=factor_cal_0, sparse_sizes=torch.Size([N, N])).cuda()
        factor_cal_sparse_1 = SparseTensor.from_edge_index(edge_index=edges, edge_attr=factor_cal_1, sparse_sizes=torch.Size([N, N])).cuda()
        factor_res_1hop = matmul(factor_cal_sparse_1, torch.ones(N, 1).cuda(), reduce='mean')
        factor_res_2hop = matmul(factor_cal_sparse_0, factor_res_1hop, reduce='mean')

        return final_h, factor_res_2hop
        # return final_h, torch.ones(N, 1).cuda()

    def __repr__(self):
        return (
                self.__class__.__name__
                + " ("
                + str(self.in_features)
                + " -> "
                + str(self.out_features)
                + ")"
        )


class GPSGCN(nn.Module):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--hidden-size", type=int, default=128)
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
            args.layers,
            args.N
        )

    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, layers, N):
        """Sparse version of GAT."""
        super(GPSGCN, self).__init__()
        self.dropout = dropout
        self.layers = layers    # 层数
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.nclass = nclass

        self.attention_size = 16
        self.alpha = alpha
        self.N = N

        self.linear_before = nn.Linear(nfeat, nhid)
        nn.init.xavier_normal_(self.linear_before.weight, gain=1.414)
        nn.init.constant_(self.linear_before.bias, 0)

        for i in range(layers):
            self.convs.append(GPSDepthAttentionLayer(
                nhid, nhid, N=self.N, attention_size=self.attention_size,  dropout=dropout, alpha=alpha, concat=True, thickness=1, need_norm=True
            ))

        for i in range(layers-1):
            self.bns.append(torch.nn.BatchNorm1d(nhid))

        self.linear_after1 = nn.Linear(nhid, nhid)
        self.linear_after2 = nn.Linear(nhid, nclass)
        nn.init.xavier_normal_(self.linear_after1.weight, gain=1.414)
        nn.init.constant_(self.linear_after1.bias, 0)
        nn.init.xavier_normal_(self.linear_after2.weight, gain=1.414)
        nn.init.constant_(self.linear_after2.bias, 0)

        self.LPWs = [nn.Parameter(torch.ones(size=(N, 1)).cuda()) for i in range(layers)]
        self.LP_alpha = 0.5

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters() # TO DO
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj, edges, degree, iftrain, label):
        # adj: sparse tensor
        x = self.linear_before(x)
        # x = F.leaky_relu(x, self.alpha)
        N = x.size()[0]

        adj_sparse_sum_rowwise = matmul(adj, torch.ones(N, 1).cuda(), reduce='sum')

        aggr_factors = torch.ones(N, 1).cuda()

        degree = torch.unsqueeze(degree, dim=1) # D^(-1/2)
        LP_degree = degree * degree
        # D ^ -1

        for layer in range(self.layers):
            new_label = self.LPWs[layer] * label
            new_label = LP_degree * matmul(adj, new_label)
            new_label = self.LP_alpha * label + (1 - self.LP_alpha) * new_label
            new_label_sum = torch.sum(new_label, dim=1).unsqueeze(1) + 1e-10
            new_label = torch.div(new_label, new_label_sum)
            label = new_label

            x, aggr_factors_tmp = \
                self.convs[layer](x, adj, aggr_factors, edges, adj_sparse_sum_rowwise, degree, iftrain, self.LPWs[layer])
            aggr_factors = aggr_factors * aggr_factors_tmp
            if layer != self.layers-1:
                x = self.bns[layer](x)
                x = F.leaky_relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # 把不同head的输出stack起来，并求和
        x = self.linear_after1(x)
        x = F.leaky_relu(x, self.alpha)
        x = self.linear_after2(x)

        return F.log_softmax(x, dim=1), label

    def loss(self, datax, datay, train_mask, adj_sparse, edges, degree, label, label_train_x, label_train_y):
        masked_label = torch.zeros(label.shape).cuda()
        masked_label = masked_label.index_fill(0, label_train_x, 1)
        masked_label = label * masked_label
        pre_x, new_label = self.forward(datax, adj_sparse, edges, degree, True, masked_label)
        # print("1", F.nll_loss(
        #     pre_x[train_mask],
        #     datay[train_mask]
        # ) )
        # print("2", F.nll_loss(
        #     new_label[label_train_y],
        #     datay[label_train_y]
        # )
        # print("loss_feature: ", F.nll_loss(
        #     pre_x[train_mask],
        #     datay[train_mask]))
        # print("loss label: ", F.nll_loss(
        #     new_label[label_train_y],
        #     datay[label_train_y]
        # ))
        # print(pre_x[train_mask][1, ])

        return F.nll_loss(
            pre_x[train_mask],
            datay[train_mask]
        ) + F.nll_loss(
            new_label[label_train_y],
            datay[label_train_y]
        )

        # return F.nll_loss(
        #     pre_x[train_mask],
        #     datay[train_mask]
        # )

    def predict(self, datax, adj_sparse, edges, degree, label):
        return self.forward(datax, adj_sparse, edges, degree, False, label)[0]
