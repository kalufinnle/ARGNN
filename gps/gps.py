import torch
import torch.nn as nn
import torch.nn.functional as F
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


class GPSAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True,thickness=1, max_dis=3, k=100, attention_size = 16, GPU = False, need_norm = True):
        super(GPSAttentionLayer, self).__init__()
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

        self.W = nn.Parameter(torch.zeros(size=(thickness, in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
        self.B = nn.Parameter(torch.zeros(size=(thickness, 1, out_features)))
        nn.init.xavier_normal_(self.B.data, gain=1.414)

        self.Wk = nn.Parameter(torch.zeros(size=(in_features, self.attention_hid)))
        nn.init.xavier_normal_(self.Wk, gain=1.414)
        self.Bk = nn.Parameter(torch.zeros(size=(1, self.attention_hid)))
        nn.init.xavier_normal_(self.Bk, gain=1.414)

        self.Wq = nn.Parameter(torch.zeros(size=(in_features, self.attention_hid)))
        nn.init.xavier_normal_(self.Wq, gain=1.414)
        self.Bq = nn.Parameter(torch.zeros(size=(1, self.attention_hid)))
        nn.init.xavier_normal_(self.Bq, gain=1.414)

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

    def forward(self, input, receptive_field, adj):
        # --------------------------------------------------------------------------------
        # 实现的逻辑
        # 在第m层，感受野中有m*num_recep个点，其中1, 2, .., m阶邻居各num_recep个
        # 先对自身做线性变换。再对感受野中的每个点算attention，选取top k, 做线性变换，加权聚合到自身上
        # 为了对不同阶邻居有所区分，对感受野中的不同阶邻居，线性变换的W不同
        # 再对感受野进行拓展。计算所有m+1阶邻居的attention，选取 top (num_recep)，加入感受野中。
        # --------------------------------------------------------------------------------
        N = input.size()[0]         # 此时N为实际的N+1(多了一个special_id)
        #test!!
        Key = torch.mm(input, self.Wk) + self.Bk
        Query = torch.mm(input, self.Wq) + self.Bq
        num_degree = adj.size()[1]  # 每个点的邻居数
        num_recep = adj.size()[1]   # 每种距离下，在感受野中的点的个数
        k = min(self.k, num_recep)  # 每种距离下，选attention为top k的点进行聚合

        # 对h做线性变换
        new_h = torch.mm(input, self.W[0]) + self.B[0]
        final_h = new_h

        # 点a作为m阶邻居聚合到点b上时，对于不同的m，做的线性变换不同
        new_h_as_nei = new_h.unsqueeze(0)
        for ii in range(1, self.thickness):
            new_h_as_nei = torch.cat((new_h_as_nei, (torch.mm(input, self.W[ii]) + self.B[ii]).unsqueeze(0)), 0)

        # 先算出simple_attention的la和ra，为后续做准备
        simple_attention_la = torch.mm(new_h, self.la_simple).view(-1) + self.Bla_simple
        simple_attention_ra = torch.mm(new_h, self.ra_simple).view(-1) + self.Bra_simple
        F.dropout(simple_attention_la, self.attention_dropout, training=self.training)
        F.dropout(simple_attention_ra, self.attention_dropout, training=self.training)


        # 聚合
        for ii in range(1):
        # for ii in range(self.thickness):
            # 算attention
            receptive_field_1layer = receptive_field[ii]
            receptive_field_1d = receptive_field_1layer.view(1, -1).squeeze(0)
            # vector_key = Key.repeat(1, num_recep).view(N*num_recep, -1)
            # vector_key = F.dropout(vector_key, self.attention_dropout, training=self.training)
            # receptive_field_1d_view = receptive_field_1d.view(1, -1).squeeze(0)
            # vector_query = torch.index_select(Query, 0, receptive_field_1d_view)
            # vector_query = F.dropout(vector_query, self.attention_dropout, training=self.training)
            # recep_attention = (vector_key * vector_query).sum(1)
            vector_key = Key.view(N, 1, -1)
            vector_key = F.dropout(vector_key, self.attention_dropout, training=self.training)
            receptive_field_1d_view = receptive_field_1d.view(1, -1).squeeze(0)
            vector_query = torch.index_select(Query, 0, receptive_field_1d_view).view(N, num_recep, -1)
            # vector_query = Query[receptive_field_1d_view].view(N, num_recep, -1)
            vector_query = F.dropout(vector_query, self.attention_dropout, training=self.training)
            recep_attention = (vector_key * vector_query).view(N*num_recep, -1).sum(1)
            recep_attention = self.leakyrelu(recep_attention)

            #test!
            # receptive_field_1layer = receptive_field[ii]
            # receptive_field_1d_view = receptive_field_1layer.view(-1)
            # # 计算GAT版本的attention
            # simple_attention_dest = torch.index_select(simple_attention_la, 0, receptive_field_1d_view)
            # simple_attention_src = simple_attention_ra.view(N, -1).repeat(1, num_recep).view(-1)
            # simple_attention = simple_attention_dest + simple_attention_src
            # simple_attention = self.leakyrelu(simple_attention)

            # test!!!!
            # recep_attention += simple_attention
            # recep_attention = simple_attention

            # 将编号为special_id的点的attention手动变为很小的值
            min_attention_value = torch.min(recep_attention).item()-1000
            recep_attention = recep_attention.view(N, -1)
            min_attention = (torch.ones(N, num_degree).float() * min_attention_value).cuda()
            recep_attention = torch.where(receptive_field_1layer != (N-1), recep_attention, min_attention)
            # print(recep_attention[123154])
            #test！！！
            recep_attention = F.softmax(recep_attention, dim=1)

            # 根据attention排序
            recep_attention_sorted, recep_attention_indice = torch.sort(recep_attention, dim=1, descending=True)
            # 切片，获得top k的attention值及编号
            recep_attention_sorted = recep_attention_sorted[:, :k]
            recep_attention_sorted = F.softmax(recep_attention_sorted, dim=1)
            recep_attention_indice = recep_attention_indice[:, :k]
            tmp_delta = torch.arange(0, N * num_recep, num_recep)
            if self.GPU:
                tmp_delta = tmp_delta.cuda()
            tmp_delta = tmp_delta.unsqueeze(1).repeat(1, num_recep)
            recep_attention_indice = (recep_attention_indice + tmp_delta).view(1, -1).squeeze(
                0)  # 将recep_field变成一行后的index
            recep_chosed_nodeid = torch.index_select(receptive_field_1d, 0, recep_attention_indice)

            # 获得top k的向量表示
            recep_chosed_new_h = torch.index_select(new_h_as_nei[ii:ii + 1, :, :].squeeze(0), 0, recep_chosed_nodeid)
            recep_chosed_new_h = recep_chosed_new_h.view(N, k, -1)  # N*k*out_feature
            # recep_attention_sorted = recep_attention_sorted.unsqueeze(2).repeat(1, 1,
            #                                                                     self.out_features)  # N*k*out_feature
            recep_attention_sorted = recep_attention_sorted.unsqueeze(2)  # N*k*out_feature
            # 聚合（即使special id被选进去了，由于其向量表示恒为全0，不会影响结果）
            final_h = torch.sum((recep_attention_sorted * recep_chosed_new_h), 1) + final_h
            # test_recep_new_h = torch.index_select(new_h_as_nei[ii], 0, receptive_field_1d_view).view(N*num_recep, -1)
            # final_h = torch.sum((recep_attention.view(-1, 1) * test_recep_new_h).view(N, num_recep, -1), 1) + final_h

        if self.need_norm:
            # batch normalization
            final_h = self.bn(final_h)
        # test
        if self.thickness != 3:
            # 若不是最后一层
            final_h = F.relu(final_h, inplace=True)
            final_h = F.dropout(final_h, self.dropout, training=self.training)

        # # 拓展感受野
        receptive_field_1d = receptive_field[self.thickness - 1:self.thickness, :, :].squeeze(0).view(1, -1).squeeze(0) # N*num_recep
        # 得到所有的k+1阶邻居
        neighbor = torch.index_select(adj, 0, receptive_field_1d).view(1, -1).squeeze(0) # N*numrecep*num_degree
        # 计算所有k+1阶邻居的attention
        # vector_key = Key.repeat(1, num_recep*num_degree).view(N * num_recep * num_degree, -1)
        # vector_query = torch.index_select(Query, 0, neighbor)
        # nei_attention = (vector_key * vector_query).sum(1)
        vector_la = torch.index_select(simple_attention_la, 0, neighbor)
        vector_ra = simple_attention_ra.view(-1, 1).repeat(1, num_recep*num_degree).view(-1)
        nei_attention = vector_la + vector_ra

        nei_attention = nei_attention.view(N, -1)  # N*(num_degree*num_recep), recep各点的attention
        neighbor = neighbor.view(N, -1)
        # 手动将special_id的点的attention置为最小
        min_attention_value = torch.min(nei_attention).item() - 1
        min_attention = (torch.ones(N, num_degree*num_recep).float() * min_attention_value).cuda()
        nei_attention = torch.where(neighbor != (N - 1), nei_attention, min_attention)
        #排序
        nei_attention_sorted, nei_attention_indice = torch.sort(nei_attention, dim=1, descending=True)
        # 切片，获得前num_recep个
        nei_attention_indice = nei_attention_indice[:, :num_recep]
        tmp_delta = torch.arange(0, N * num_recep * num_degree, num_recep * num_degree)
        if self.GPU:
            tmp_delta =tmp_delta.cuda()
        tmp_delta = tmp_delta.unsqueeze(1).repeat(1,num_recep)

        nei_attention_indice = (nei_attention_indice + tmp_delta).view(1, -1).squeeze(0)
        expand_chosed_nodeid = torch.index_select(neighbor.view(1, -1).squeeze(0), 0, nei_attention_indice)
        expand_chosed_nodeid = expand_chosed_nodeid.view(N, -1)
        expand_chosed_nodeid = expand_chosed_nodeid.unsqueeze(0)
        receptive_field = torch.cat((receptive_field, expand_chosed_nodeid), 0)

        return final_h, receptive_field

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
        )

    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, layers=2,):
        """Sparse version of GAT."""
        super(GPS, self).__init__()
        self.GPU = True
        self.dropout = dropout
        self.heads = nheads     # head: 感受野不同，最后一层cat起来
        self.layers = layers    # 层数
        self.attentions = []
        self.subheads = 4       # subhead: 同一感受野内部，attention的W矩阵不同
        self.nclass = nclass
        self.attention_size = 32

        # 第一层： in: nfeat; out: nhid，即参数中的hidden-size
        self.attentions.append([
            [GPSAttentionLayer(
                nfeat, nhid, attention_size=self.attention_size,  dropout=dropout, alpha=alpha, concat=True, thickness=1, GPU=self.GPU, need_norm=True
            ) for __ in range(self.subheads)]
            for _ in range(nheads)
        ])

        # 第2-(n-1)层 in和out均为: nhid * 每种感受野内部的subhead数
        for i in range(layers - 2):
            self.attentions.append([
                [GPSAttentionLayer(
                    nhid * self.subheads, nhid, attention_size=self.attention_size, dropout=dropout, alpha=alpha, concat=True, thickness=i + 2, GPU=self.GPU, need_norm=True
                ) for __ in range(self.subheads)]
                for _ in range(nheads)
            ])

        # 第n层： in: nhid * 每种感受野内部的subhead数; out: nclass,即类别数
        self.attentions.append([
            [GPSAttentionLayer(
                nhid * self.subheads, nclass, attention_size=self.attention_size, dropout=dropout, alpha=alpha, concat=True, thickness=layers, GPU=self.GPU, need_norm=False
            ) for __ in range(self.subheads)]
            for _ in range(nheads)
        ])

        for i, attention in enumerate(self.attentions):
            for j, att in enumerate(attention):
                for t, at in enumerate(att):
                    self.add_module("attention_{}_{}_{}".format(i, j, t), at)

    def forward(self, x, adj, max_degree):

        # adj 采样后的邻接表
        # N0总点数

        N0 = x.size()[0]
        # 增加序号为special_id = N0的特殊节点，代表空节点。若邻居为special_id，则表明该邻居为空该邻居为空。
        special_id = N0
        inputfeature_size = x.size()[1]

        # 在x的最后加一行，作为special_id的向量表示
        x_append = torch.zeros(1, inputfeature_size).float()
        if self.GPU:
            x_append = x_append.cuda()
        x = torch.cat((x, x_append), 0)
        # 在adj的最后加一行，全为special_id，作为special_id的邻居
        adj_append = (torch.ones(1, max_degree) * special_id).long()
        if self.GPU:
            adj_append = adj_append.cuda()
        adj = torch.cat((adj.long(), adj_append), 0)

        y = [x.clone() for _ in range(self.heads)]
        z = [x.clone() for _ in range(self.heads)]
        receptive_field = [adj.unsqueeze(0) for _ in range(self.heads)]
        for layer in range(self.layers):
            # head: 感受野不同，最后一层cat起来
            for number_attention in range(self.heads):
                # subhead: 共享感受野，但attention的W矩阵不同
                for number_subattention in range(self.subheads):
                    ytmp, receptive_field[number_attention] = \
                        self.attentions[layer][number_attention][number_subattention](y[number_attention],
                                                                                      receptive_field[number_attention],
                                                                                      adj)
                    if number_subattention == 0:
                        z[number_attention] = ytmp
                    else:
                        z[number_attention] = torch.cat((z[number_attention], ytmp), 1)
                y[number_attention] = z[number_attention]

        # print("attention", receptive_field[0][:, 119, :])
        # 把不同head的输出stack起来，并求和
        x = torch.stack(y)
        x = x.sum(0)
        x = x[:N0, :]   # 需要去掉special_id的输出
        return F.log_softmax(x, dim=1)

    def loss(self, datax, datay, train_mask, adj_processed, max_degree):

        return F.nll_loss(
            self.forward(datax, adj_processed, max_degree)[train_mask],
            datay[train_mask],
        )

    def predict(self, datax, adj_processed, max_degree):
        return self.forward(datax, adj_processed, max_degree)