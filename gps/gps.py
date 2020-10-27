import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import spmm

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


class GPSAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, num_heads, attn_drop, atten_hid=32):
        super(GPSAttentionLayer, self).__init__()
        self.SpecialSpmm = SpecialSpmm()
        gain = 1.414
        self.fc = nn.Linear(in_features, out_features*num_heads, bias=False)
        nn.init.xavier_normal_(self.fc.weight, gain=gain)

        self.attn_drop = nn.Dropout(attn_drop)

        self.attn_fc0 = nn.Linear(in_features, atten_hid)
        nn.init.xavier_normal_(self.attn_fc0.weight, gain=gain)
        self.attn_fc1 = nn.Linear(atten_hid * 3, atten_hid)
        nn.init.xavier_normal_(self.attn_fc1.weight, gain=gain)
        self.attn_fc2 = nn.Linear(atten_hid, 1)
        nn.init.xavier_normal_(self.attn_fc2.weight, gain=gain)


    def forward(self, input:torch.Tensor, pre_edge_feat, adj, degree):
        # --------------------------------------------------------------------------------
        # 实现的逻辑
        # 在第m层，感受野中有m*num_recep个点，其中1, 2, .., m阶邻居各num_recep个
        # 先对自身做线性变换。再对感受野中的每个点算attention，选取top k, 做线性变换，加权聚合到自身上
        # 为了对不同阶邻居有所区分，对感受野中的不同阶邻居，线性变换的W不同
        # 再对感受野进行拓展。计算所有m+1阶邻居的attention，选取 top (num_recep)，加入感受野中。
        # --------------------------------------------------------------------------------
        N = input.size()[0]         # 此时N为实际的N+1(多了一个special_id)
        x = self.fc(input)
        atten_feature = self.attn_fc0(input)
        inputL = atten_feature[adj[0]]
        inputR = atten_feature[adj[1]]
        inputC = torch.abs(inputL - inputR)
        edge_feat = torch.cat((inputL, inputR, inputC),dim=1)
        edge_feat = self.attn_drop(edge_feat)
        edge_feat = self.attn_fc1(edge_feat)
        edge_feat = F.relu(edge_feat)
        edge_feat = self.attn_fc2(edge_feat)
        edge_feat = F.sigmoid(edge_feat)
        edge_feat = edge_feat.squeeze(dim=1)
        degree = degree.unsqueeze(1)
        D = torch.pow(degree, 0.5)
        new_x = torch.div(x, D)
        aggr_x = spmm(adj, torch.ones(adj.size()[1]).cuda(), N, N, new_x)
        aggr_x = torch.div(aggr_x, D)

        # print(x.size())
        # print(pre_edge_feat)
        # print("x", x, x.size())
        # print("pre_edge", pre_edge_feat, pre_edge_feat.size())
        # breakpoint()
        aggr_x = aggr_x * pre_edge_feat + x * (1 - pre_edge_feat)
        edge_feat = spmm(adj, edge_feat, N, N, torch.ones(N,1).cuda()).squeeze(1)
        edge_feat = spmm(adj, torch.ones(adj.size()[1]).cuda(), N, N, edge_feat)
        edge_feat = torch.div(edge_feat, degree)
        # print(edge_feat.size())
        return aggr_x, edge_feat


    def __repr__(self):
        return (
                self.__class__.__name__
                + " ("
                + str(self.in_features)
                + " -> "
                + str(self.out_features)
                + ")"
        )


class GPS(nn.Module):
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
            args.layers,
            args.attention_hid,
            args.attention_dropout
        )

    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, layers,attention_hid,attention_dropout):
        """Sparse version of GAT."""
        super(GPS, self).__init__()
        self.layers = layers
        self.dropout0 = torch.nn.Dropout(min(0.1, dropout))
        self.dropout = torch.nn.Dropout(dropout)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.activation = nn.functional.relu
        nheads = 1
        for i in range(layers):
            in_hidden = nheads * nhid if i > 0 else nfeat
            out_hidden = nhid if i < layers - 1 else nclass
            # in_channels = n_heads if i > 0 else 1
            out_channels = nheads
            self.convs.append(GPSAttentionLayer(in_hidden, out_hidden, num_heads=nheads, attn_drop=attention_dropout))
            if i < layers - 1:
                self.bns.append(nn.BatchNorm1d(out_channels * out_hidden))

    def forward(self, x, adj, degree):
        h = x
        h = self.dropout0(h)
        N = x.size()[0]
        edge_feat = torch.div(torch.ones((N,1)),2).cuda()
        for i in range(self.layers):
            x, edge_feat = self.convs[i](x, edge_feat, adj, degree)
            if i < self.layers-1:
                x = x.flatten(1)
                x = self.bns[i](x)
                x = self.activation(x)
                x = self.dropout(x)

        return F.log_softmax(x, dim=1)

    def loss(self, datax, datay, train_mask, adj, degree):

        return F.nll_loss(
            self.forward(datax, adj, degree)[train_mask],
            datay[train_mask]
        )

    def predict(self, datax, adj, degree):
        return self.forward(datax, adj, degree)