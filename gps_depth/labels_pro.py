import torch.nn as nn
import torch
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul
import torch.nn.functional as F


class LP(nn.Module):

    def __init__(self, n_classes, layers, N):
        """Sparse version of GAT."""
        super(LP, self).__init__()
        self.classes = n_classes
        self.layers = layers
        self.N = N

        self.W = nn.Parameter(torch.ones(size=(1, N)))
        self.alpha = 0.5
        # self.alpha = nn.Parameter(torch.ones(size=(1, 1)))

    def forward(self, x, adj, degree):
        #x N*classes
        #D -1/2
        degree = torch.unsqueeze(degree, dim=1)
        #D ^ -1
        degree = degree * degree
        for i in range(self.layers):
            new_x = self.W * x
            new_x = matmul(adj, new_x)
            new_x = degree * new_x
            new_x = self.alpha * x + (1-self.alpha) * new_x
            new_x_sum = torch.sum(new_x, dim=1)
            new_x = torch.div(new_x, new_x_sum)
            x = new_x
        return x

    def loss(self, datax, datay, train_mask, adj_sparse, degree):

        return F.nll_loss(
            self.forward(datax, adj_sparse, degree)[train_mask],
            datay[train_mask],
        )

    def predict(self, datax, adj_sparse, degree):
        return self.forward(datax, adj_sparse, degree)