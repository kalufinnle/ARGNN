import torch
import numpy as np
import copy
from ogb.nodeproppred import PygNodePropPredDataset
from tqdm import tqdm
import argparse
from gps_gcn.gps_gcn import GPSGCN
from gps_depth.labels_pro import LP
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here


class data():
    def __init__(self, node_attr, y, edges, train_idx, valid_idx, test_idx):
        """Get the information of the graph"""
        self.node_attr = node_attr
        self.y = y.squeeze(1)
        self.edges = edges
        self.num_features = node_attr.size()[1]
        self.num_classes = torch.max(self.y).item()+1 # count from 0
        self.N = node_attr.size()[0]
        self.train_mask = np.zeros(self.N, dtype=bool)
        self.valid_mask = np.zeros(self.N, dtype=bool)
        self.test_mask = np.zeros(self.N, dtype=bool)
        self.train_idx = train_idx
        self.train_mask[train_idx] = True
        self.test_mask[test_idx] = True
        self.valid_mask[valid_idx] = True
        self.train_mask = torch.from_numpy(self.train_mask)
        self.valid_mask = torch.from_numpy(self.valid_mask)
        self.test_mask = torch.from_numpy(self.test_mask)
        self.node_label = torch.nn.functional.one_hot(self.y, self.num_classes)
        degree = [1 for _ in range(self.N)]
        for i in range(self.edges.size()[1]):
            degree[self.edges[0][i]] += 1
            degree[self.edges[1][i]] += 1
        self.degree = torch.tensor(degree).float()
        self.degree = torch.pow(self.degree, -0.5)

    def apply(self, param):
        # for x in self.
        self.node_label = param(self.node_label)
        self.train_idx = param(self.train_idx)
        self.node_attr = param(self.node_attr)
        self.y = param(self.y)
        self.edges = param(self.edges)
        self.node_attr = param(self.node_attr)
        self.train_mask = param(self.train_mask)
        self.valid_mask = param(self.valid_mask)
        self.test_mask = param(self.test_mask)
        self.degree = param(self.degree)
        pass

def build_dataset_ogb(dataset):
    if dataset == "ogbn_arxiv":
        """load the dataset to class data"""
        dataset = PygNodePropPredDataset(name="ogbn-arxiv")
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph = dataset[0]
        return data(node_attr=graph.x, edges=graph.edge_index, y=graph.y, train_idx=train_idx, valid_idx=valid_idx, test_idx=test_idx)

class NodeClassification():
    """Node classification task."""
    def __init__(self, args, dataset=None, model=None):

        self.device = torch.device('cpu' if args.cpu else 'cuda')

        dataset = build_dataset_ogb(args.dataset)
        self.data = dataset
        self.data.apply(lambda x: x.to(self.device))

        args.num_features = dataset.num_features
        args.num_classes = dataset.num_classes
        args.N = dataset.node_attr.size()[0]
        self.layers = args.layers

        model = GPSGCN.build_model_from_args(args)
        self.model = model.to(self.device)

        self.patience = args.patience
        self.max_epoch = args.max_epoch

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.985)
        #sample from the adj matrix to adj list
        self.adj_sparse, self.edges = self.construct_adj(self.data.edges, self.data.node_attr.size()[0])
        if self.device == torch.device('cuda'):
            self.adj_sparse = self.adj_sparse.cuda()
            self.edges = self.edges.cuda()
        self.max_valid_acc = 0

    def train(self, writer:SummaryWriter):
        epoch_iter = tqdm(range(self.max_epoch))
        patience = 0
        best_score = 0
        best_loss = np.inf
        max_score = 0
        min_loss = np.inf
        max_valid_acc = 0
        for epoch in epoch_iter:

            self._train_step()
            train_acc, train_loss = self._test_step(split="train")


            val_acc, val_loss = self._test_step(split="val")

            writer.add_scalar('training acc',train_acc,epoch)
            writer.add_scalar('training loss',train_loss,epoch)
            writer.add_scalar('valid acc', val_acc, epoch)
            writer.add_scalar('valid loss', val_loss, epoch)

            epoch_iter.set_description(
                f"Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Val Loss:{val_loss:.4f}"
            )
            max_valid_acc = max(max_valid_acc, val_acc)
            self.max_valid_acc = max_valid_acc
            if val_loss <= min_loss or val_acc >= max_score:
                if val_loss <= best_loss:  # and val_acc >= best_score:
                    best_loss = val_loss
                    best_score = val_acc
                    best_model = copy.deepcopy(self.model)
                min_loss = np.min((min_loss, val_loss))
                max_score = np.max((max_score, val_acc))
                patience = 0
            else:
                patience += 1
                if patience == self.patience:
                    self.model = best_model
                    epoch_iter.close()
                    break
        test_acc, _ = self._test_step(split="test")
        print(f"Test accuracy = {test_acc}")
        print(f"max_valid_acc = {max_valid_acc}")
        return dict(Acc=test_acc)

    def _train_step(self):
        self.model.train()
        self.optimizer.zero_grad()
        random_mask = np.random.randint(2, size=self.data.train_idx.size()[0])
        label_train_x = torch.from_numpy(np.nonzero(random_mask)[0]).cuda()
        label_train_x = self.data.train_idx[label_train_x]
        label_train_y = torch.from_numpy(np.nonzero(1 - random_mask)[0]).cuda()
        label_train_y = self.data.train_idx[label_train_y]

        loss = self.model.loss(self.data.node_attr, self.data.y,
                               self.data.train_mask, self.adj_sparse, self.edges,
                               self.data.degree, self.data.node_label,
                               label_train_x, label_train_y)

        loss.backward()
        self.optimizer.step()
        # self.scheduler.step()

    def _test_step(self, split="val"):
        self.model.eval()
        #the result of of the model
        logits = self.model.predict(self.data.node_attr, self.adj_sparse, self.edges, self.data.degree, self.data.node_label)
        if split == "train":
            mask = self.data.train_mask
        elif split == "val":
            mask = self.data.valid_mask
        else:
            mask = self.data.test_mask

        loss = F.nll_loss(logits[mask], self.data.y[mask])
        # if self.max_valid_acc > 0.69:
        #     breakpoint()

        #the predict result
        pred = logits[mask].max(1)[1]
        acc = pred.eq(self.data.y[mask]).sum().item() / mask.sum().item()
        return acc, loss.item()

    def construct_x(self):
        x = self.data.node_attr
        return x


    def construct_adj(self, adj_list, N):
        # 把自环去掉！！
        '''
        sample from the adj matrix to adj list, every node has max_degree neighbors 
        node N is the additional node which do not exist in the graph to satisfy degree requirement     
        every node has a self-loop
        '''
        tmp_edges = adj_list.cpu().numpy().T.tolist()
        bi_edge = []
        bi_edge_cnt = 0

        for edge in tmp_edges:
            bi_edge.append([edge[0], edge[1]])
            bi_edge.append([edge[1], edge[0]])
            bi_edge_cnt += 2

        for n in range(N):
            bi_edge.append([n, n])

        bi_edge = torch.LongTensor(bi_edge).t()

        return SparseTensor.from_edge_index(edge_index=bi_edge, sparse_sizes=torch.Size([N, N])), bi_edge


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="gps args")  # fmt: off

    parser.add_argument("--layers", type=int, default=5, help='the layers number')
    parser.add_argument("--dataset", type=str, default="ogbn_arxiv", help='chose a dataset')
    parser.add_argument('--cpu', action='store_true', help='use CPU instead of CUDA')
    parser.add_argument('--max-epoch', default=2000, type=int)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--weight-decay', default=0, type=float)
    parser.add_argument('--device-id', default=[0], type=int, nargs='+',
                        help='which GPU to use')
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.2, help='alpha in leakyrelu')
    parser.add_argument("--nheads", type=int, default=1, help='head number')
    parser.add_argument("--attention_hid", type=int, default=1, help='attention head number')
    parser.add_argument("--max-degree", type=int, default=10)
    parser.add_argument("--name", type=str, help="tensorboard name")



    args = parser.parse_args()
    writer = SummaryWriter(f"runs/{args.name}")

    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id[0])
    my_model = NodeClassification(args)
    my_model.train(writer)
    # fmt: on
    pass
