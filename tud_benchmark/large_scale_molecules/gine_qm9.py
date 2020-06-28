import sys

sys.path.insert(0, '..')
sys.path.insert(0, '../..')
sys.path.insert(0, '.')

import os.path as osp
import numpy as np
from torch.nn import Linear as Lin
from torch.nn import Sequential, Linear, ReLU
import torch_geometric.transforms as T

from torch_geometric.data import DataLoader
from torch_geometric.datasets import QM9
from torch_geometric.utils import remove_self_loops
import torch
from torch_geometric.nn import MessagePassing, Set2Set
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GINConv(MessagePassing):
    def __init__(self, emb_dim, dim1, dim2):
        super(GINConv, self).__init__(aggr="add")

        self.bond_encoder = Sequential(Linear(emb_dim, dim1), torch.nn.BatchNorm1d(dim1), ReLU(), Linear(dim1, dim1),
                                       torch.nn.BatchNorm1d(dim1), ReLU())

        self.mlp = Sequential(Linear(dim1, dim2), torch.nn.BatchNorm1d(dim2), ReLU(), Linear(dim2, dim2),
                              torch.nn.BatchNorm1d(dim2), ReLU())

        self.eps = torch.nn.Parameter(torch.Tensor([0]))

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)

        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class NetGINE(torch.nn.Module):
    def __init__(self, dim):
        super(NetGINE, self).__init__()

        num_features = 13
        dim = dim

        self.conv1 = GINConv(6, num_features, dim)
        self.conv2 = GINConv(6, dim, dim)
        self.conv3 = GINConv(6, dim, dim)
        self.conv4 = GINConv(6, dim, dim)
        self.conv5 = GINConv(6, dim, dim)
        self.conv6 = GINConv(6, dim, dim)

        self.set2set = Set2Set(1 * dim, processing_steps=6)

        self.fc1 = Lin(2 * dim, dim)
        self.fc4 = Linear(dim, 12)

    def forward(self, data):
        x = data.x

        x_1 = F.relu(self.conv1(x, data.edge_index, data.edge_attr))
        x_2 = F.relu(self.conv2(x_1, data.edge_index, data.edge_attr))
        x_3 = F.relu(self.conv3(x_2, data.edge_index, data.edge_attr))
        x_4 = F.relu(self.conv4(x_3, data.edge_index, data.edge_attr))
        x_5 = F.relu(self.conv5(x_4, data.edge_index, data.edge_attr))
        x_6 = F.relu(self.conv6(x_5, data.edge_index, data.edge_attr))
        x = x_6

        x = self.set2set(x, data.batch)

        x = F.relu(self.fc1(x))
        x = self.fc4(x)
        return x


class Complete(object):
    def __call__(self, data):
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data


results = []
results_log = []
for _ in range(5):

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'datasets', 'QM9')
    dataset = QM9(path, transform=T.Compose([Complete(), T.Distance(norm=False)]))
    dataset.data.y = dataset.data.y[:, 0:12]

    dataset = dataset.shuffle()

    tenpercent = int(len(dataset) * 0.1)
    mean = dataset.data.y.mean(dim=0, keepdim=True)
    std = dataset.data.y.std(dim=0, keepdim=True)
    dataset.data.y = (dataset.data.y - mean) / std
    mean, std = mean.to(device), std.to(device)

    tenpercent = int(len(dataset) * 0.1)
    test_dataset = dataset[:tenpercent].shuffle()
    val_dataset = dataset[tenpercent:2 * tenpercent].shuffle()
    train_dataset = dataset[2 * tenpercent:].shuffle()

    print(len(train_dataset), len(val_dataset), len(test_dataset))

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = NetGINE(64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.5, patience=5,
                                                           min_lr=0.0000001)


    def train():
        model.train()
        loss_all = 0

        lf = torch.nn.L1Loss()

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            loss = lf(model(data), data.y)

            loss.backward()
            loss_all += loss.item() * data.num_graphs
            optimizer.step()
        return (loss_all / len(train_loader.dataset))


    @torch.no_grad()
    def test(loader):
        model.eval()
        error = torch.zeros([1, 12]).to(device)

        for data in loader:
            data = data.to(device)
            error += ((data.y * std - model(data) * std).abs() / std).sum(dim=0)

        error = error / len(loader.dataset)
        error_log = torch.log(error)

        return error.mean().item(), error_log.mean().item()


    test_error = None
    test_error_log = None
    best_val_error = None
    for epoch in range(1, 1001):
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss = train()
        val_error, _ = test(val_loader)
        scheduler.step(val_error)

        if best_val_error is None or val_error <= best_val_error:
            test_error, test_error_log = test(test_loader)
            best_val_error = val_error

        print('Epoch: {:03d}, LR: {:.7f}, Loss: {:.7f}, Validation MAE: {:.7f}, '
              'Test MAE: {:.7f}, Test MAE: {:.7f}'.format(epoch, lr, loss, val_error, test_error, test_error_log))

        if lr < 0.000001:
            print("Converged.")
            break

    results.append(test_error)
    results_log.append(test_error_log)

print("########################")
print(results)
results = np.array(results)
print(results.mean(), results.std())

print(results_log)
results_log = np.array(results_log)
print(results_log.mean(), results_log.std())
