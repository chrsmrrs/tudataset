import sys

sys.path.insert(0, '..')
sys.path.insert(0, '.')

import os.path as osp
import numpy as np
import torch
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_mean_pool, GINConv, MessagePassing
from torch_geometric.datasets import TUDataset

from torch_geometric.data import DataLoader
import torch.nn.functional as F


class GINConv(MessagePassing):
    def __init__(self, emb_dim, dim1, dim2):
        super(GINConv, self).__init__(aggr="add")

        self.bond_encoder = Sequential(Linear(emb_dim, dim1), ReLU(), Linear(dim1, dim1))
        self.mlp = Sequential(Linear(dim1, dim1), ReLU(), Linear(dim1, dim2))
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

        num_features = 3
        dim = dim

        self.conv1 = GINConv(num_features, 28, 256)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        self.conv2 = GINConv(num_features, 256, 256)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        self.conv3 = GINConv(num_features, 256, 256)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        self.conv4 = GINConv(num_features, 256, 256)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(4 * dim, dim)
        self.fc2 = Linear(dim, dim)
        self.fc3 = Linear(dim, dim)
        self.fc4 = Linear(dim, 1)

    def forward(self, data):
        x = data.x

        x_1 = F.relu(self.conv1(x, data.edge_index, data.edge_attr))
        x_1 = self.bn1(x_1)
        x_2 = F.relu(self.conv2(x_1, data.edge_index, data.edge_attr))
        x_2 = self.bn2(x_2)
        x_3 = F.relu(self.conv3(x_2, data.edge_index, data.edge_attr))
        x_3 = self.bn3(x_3)
        x_4 = F.relu(self.conv3(x_3, data.edge_index, data.edge_attr))
        x_4 = self.bn4(x_4)

        x = torch.cat([x_1, x_2, x_3, x_4], dim=-1)
        x = global_mean_pool(x, data.batch)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x.view(-1)


results = []
for _ in range(5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'datasets', 'ZINC')
    dataset = TUDataset(path, name="ZINC_full")

    train_dataset = dataset[0:220011].shuffle()
    val_dataset = dataset[225011:249456].shuffle()
    test_dataset = dataset[220011:225011].shuffle()

    batch_size = 25
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = NetGINE(256).to(device)
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


    def test(loader):
        model.eval()
        error = 0

        for data in loader:
            data = data.to(device)
            error += (model(data) - data.y).abs().sum().item()
        return error / len(loader.dataset)


    best_val_error = None
    for epoch in range(1, 1001):
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss = train()
        val_error = test(val_loader)
        scheduler.step(val_error)

        if best_val_error is None or val_error <= best_val_error:
            test_error = test(test_loader)
            best_val_error = val_error

        print('Epoch: {:03d}, LR: {:.7f}, Loss: {:.7f}, Validation MAE: {:.7f}, '
              'Test MAE: {:.7f}'.format(epoch, lr, loss, val_error, test_error))

        if lr < 0.000001:
            print("Converged.")
            break

    results.append(test_error)

print("########################")
print(results)
results = np.array(results)
print(results.mean(), results.std())
