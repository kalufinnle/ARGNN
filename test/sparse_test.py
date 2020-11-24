import torch
from torch_sparse import SparseTensor, matmul

if __name__ == "__main__":
    N = 3
    edges = [[0, 0], [0, 1], [1, 2], [2, 0], [2, 2]]
    edges = torch.LongTensor(edges).t()
    a_edge = [1, 2, 5, 8, 14]
    a_edge = torch.Tensor(a_edge)
    a_sparse = SparseTensor.from_edge_index(edge_index=edges, edge_attr=a_edge, sparse_sizes=torch.Size([N, N])).cuda()
    print(a_sparse)
    a_sparse_sum_rowwise = matmul(a_sparse, torch.ones(N, 1).cuda(), reduce='sum')
    print('sum', a_sparse_sum_rowwise)
    a_sparse_mean_rowwise = matmul(a_sparse, torch.ones(N, 1).cuda(), reduce='mean')
    print('mean', a_sparse_mean_rowwise)
