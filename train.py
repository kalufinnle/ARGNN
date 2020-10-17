import torch
import numpy as np
import copy
from ogb.nodeproppred import PygNodePropPredDataset
from tqdm import tqdm
import argparse
from gps.gps import GPS
import copy
import torch.nn.functional as F


class data():
    def __init__(self, node_attr, y, edges, train_idx, valid_idx, test_idx):
        """Get the information of the graph"""
        self.node_attr = node_attr
        self.y = y.squeeze(1)
        self.edges = edges
        self.num_features = node_attr.size()[1]
        self.num_classes = torch.max(self.y).item() + 1  # count from 0
        self.N = node_attr.size()[0]
        self.train_mask = np.zeros(self.N, dtype=bool)
        self.valid_mask = np.zeros(self.N, dtype=bool)
        self.test_mask = np.zeros(self.N, dtype=bool)
        self.train_mask[train_idx] = True
        self.test_mask[test_idx] = True
        self.valid_mask[valid_idx] = True
        self.train_mask = torch.from_numpy(self.train_mask)
        self.valid_mask = torch.from_numpy(self.valid_mask)
        self.test_mask = torch.from_numpy(self.test_mask)

    def apply(self, param):
        # for x in self.
        self.node_attr = param(self.node_attr)
        self.y = param(self.y)
        self.edges = param(self.edges)
        self.node_attr = param(self.node_attr)
        self.train_mask = param(self.train_mask)
        self.valid_mask = param(self.valid_mask)
        self.test_mask = param(self.test_mask)
        pass


def build_dataset_ogb(dataset):
    if dataset == "ogbn_arxiv":
        """load the dataset to class data"""
        dataset = PygNodePropPredDataset(name="ogbn-arxiv")
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph = dataset[0]
        return data(node_attr=graph.x, edges=graph.edge_index, y=graph.y, train_idx=train_idx, valid_idx=valid_idx,
                    test_idx=test_idx)


class NodeClassification():
    """Node classification task."""

    def __init__(self, args, dataset=None, model=None):

        self.device = torch.device('cpu' if args.cpu else 'cuda')

        dataset = build_dataset_ogb(args.dataset)
        self.data = dataset
        self.data.apply(lambda x: x.to(self.device))

        args.num_features = dataset.num_features
        args.num_classes = dataset.num_classes

        model = GPS.build_model_from_args(args)
        self.model = model.to(self.device)
        self.patience = args.patience
        self.max_epoch = args.max_epoch

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.98)
        # the num of neighbors for one node
        self.max_degree = args.max_degree
        # sample from the adj matrix to adj list
        self.adj_processed = self.construct_adj(self.data.edges, self.data.node_attr.size()[0])
        if self.device == torch.device('cuda'):
            self.adj_processed = self.adj_processed.cuda()

    def train(self):
        epoch_iter = tqdm(range(self.max_epoch))
        patience = 0
        best_score = 0
        best_loss = np.inf
        max_score = 0
        min_loss = np.inf
        max_valid_acc = 0
        for epoch in epoch_iter:
            self._train_step()
            train_acc, _ = self._test_step(split="train")
            val_acc, val_loss = self._test_step(split="val")
            epoch_iter.set_description(
                f"Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Val Loss:{val_loss:.4f}"
            )
            #debug
            if val_acc > 0.69:
                breakpoint()

            max_valid_acc = max(max_valid_acc, val_acc)
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

        loss = self.model.loss(self.data.node_attr, self.data.y, self.data.train_mask, self.adj_processed,
                               self.max_degree)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

    def _test_step(self, split="val"):
        self.model.eval()
        # the result of of the model
        logits = self.model.predict(self.data.node_attr, self.adj_processed, self.max_degree)
        if split == "train":
            mask = self.data.train_mask
        elif split == "val":
            mask = self.data.valid_mask
        else:
            mask = self.data.test_mask

        loss = F.nll_loss(logits[mask], self.data.y[mask])
        # the predict result
        # breakpoint()
        pred = logits[mask].max(1)[1]
        acc = pred.eq(self.data.y[mask]).sum().item() / mask.sum().item()

        # devbug
        if acc > 0.69 and split=="train":
            print("asd")
            self.draw_pict(self.neighbor, self.data.valid_mask, logits.max(1)[1])

        return acc, loss.item()

    def construct_x(self):
        x = self.data.node_attr
        return x

    def draw_pict(self, nei, mask, pred):
        mx = 0
        for i in nei:
            mx = max(mx, len(i))
        mx = 30
        x = list(range(0, mx+1))
        y = [0]*(mx+1)
        y2 = [0]*(mx+1)
        y3 = [0] * (mx + 1)
        y_valid = [0] * (mx + 1)


        total_sum = 0
        tmp_y = self.data.y.cpu()
        for i, nei_i in enumerate(nei):
            # print(i)
            cls = tmp_y[i]

            if mask[i] == True:
                # print(len(nei_i))
                if len(nei_i) <= mx:
                    y_valid[len(nei_i)] += 1
                    if cls == pred[i]:
                        y3[len(nei_i)] += 1

            sum = 0
            # for j in nei_i:
            #     if tmp_y[j] == cls:
            #         sum += 1
            sum = sum/len(nei_i)
            total_sum += sum
            if len(nei_i) <= mx:
                y[len(nei_i)] += 1
                y2[len(nei_i)] += sum
            else:
                y[mx] += 1
                y2[mx] += sum


        for i in range(mx+1):
            if y[i]==0:
                continue
            y2[i]/=y[i]
            if y_valid[i]==0:
                continue
            y3[i]/=y_valid[i]
        total_sum /= len(nei)
        print(f"total_sum:{total_sum}")

        import matplotlib.pyplot as plt
        # plt.plot(x,y_valid)
        # plt.savefig("tmp.jpg")
        # while True:
        #     print("ha")

        # plt.plot(x, y2)
        # plt.savefig("tmp2.jpg")

        plt.plot(x, y3)
        plt.savefig("tmp3.jpg")




    def construct_adj(self, adj_list, N):
        '''
        sample from the adj matrix to adj list, every node has max_degree neighbors
        node N is the additional node which do not exist in the graph to satisfy degree requirement
        every node has a self-loop
        '''
        tmp_edges = adj_list.cpu().numpy().T.tolist()
        neighbor = [[] for i in range(N)]

        for edge in tmp_edges:
            neighbor[edge[0]].append(edge[1])
            neighbor[edge[1]].append(edge[0])

        #debug
        self.neighbor = copy.deepcopy(neighbor)

        adj = np.ones((N, self.max_degree), dtype=np.long)
        for nodeid in range(N):
            neighbors = neighbor[nodeid]

            if len(neighbors) == 0:
                neighbors = np.ones(self.max_degree, dtype=np.long) * N
            elif len(neighbors) >= self.max_degree:
                neighbors = np.array(neighbors)
                # sampling
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                # add virtual node instead of upsampling
                for i in range(len(neighbors), self.max_degree):
                    neighbors.append(N)
                neighbors = np.array(neighbors)
            adj[nodeid, :] = neighbors

        adj = torch.from_numpy(adj).int()
        return adj


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="gps args")  # fmt: off

    parser.add_argument("--layers", type=int, default=2, help='the layers number')
    parser.add_argument("--dataset", type=str, default="ogbn_arxiv", help='chose a dataset')
    parser.add_argument('--cpu', action='store_true', help='use CPU instead of CUDA')
    parser.add_argument('--max-epoch', default=1000, type=int)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--weight-decay', default=0, type=float)
    parser.add_argument('--device-id', default=[0], type=int, nargs='+',
                        help='which GPU to use')
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.2, help='alpha in leakyrelu')
    parser.add_argument("--nheads", type=int, default=1, help='head number')
    parser.add_argument("--attention_hid", type=int, default=1, help='attention head number')
    parser.add_argument("--max-degree", type=int, default=10)

    args = parser.parse_args()
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id[0])
    my_model = NodeClassification(args)
    my_model.train()
    # fmt: on
    pass
