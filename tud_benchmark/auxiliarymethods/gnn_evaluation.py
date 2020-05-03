from __future__ import division

import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset


# Return arg max of iterable, e.g., a list.
def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]



# One training epoch for GNN model.
def train(train_loader, model, optimizer, device):
    model.train()

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        optimizer.step()


# Get acc. of GNN model.
def test(loader, model, device):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


# Train GNN model.
def train_model(train_loader, val_loader, test_loader, model, optimizer, scheduler, device, num_epochs):
    test_error = None
    best_val_error = None

    for epoch in range(1, num_epochs + 1):
        lr = scheduler.optimizer.param_groups[0]['lr']
        train(train_loader, model, optimizer, device)
        val_error = test(val_loader, model, device)
        scheduler.step(val_error)

        if best_val_error is None or val_error <= best_val_error:
            test_error = test(test_loader, model, device)
            best_val_error = val_error

        if lr < 0.000001:
            break

    return best_val_error, test_error


# 10-CV for GNN training and hyperparameter selection.
def gnn_evaluation(gnn, ds_name, layers, hidden, max_num_epochs=100, batch_size=25, start_lr=0.001, num_repetitions=10,
                   all_std=False):
    # Load dataset.
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'datasets', ds_name)
    dataset = TUDataset(path, name=ds_name).shuffle()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_accuracies_all = []
    test_accuracies_complete = []

    for i in range(num_repetitions):
        kf = KFold(n_splits=10, shuffle=True)
        # Test acc. over all folds.
        test_accuracies = []

        for train_index, test_index in kf.split(list(range(len(dataset)))):
            # Sample 10% split from training split for validation.
            train_index, val_index = train_test_split(train_index, test_size=0.1)

            test_dataset = dataset[test_index.tolist()]
            val_dataset = dataset[val_index.tolist()]
            train_dataset = dataset[train_index.tolist()]
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            train_loader = DataLoader(train_dataset, batch_size=batch_size)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

            # Collect val. and test acc. over all hyperparameter combinations.
            vals = []
            tests = []
            for l in layers:
                for h in hidden:
                    model = gnn(dataset, l, h).to(device)
                    # Reset parameters before each run.
                    model.reset_parameters()

                    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                           factor=0.5, patience=5,
                                                                           min_lr=0.0000001)

                    val_acc, test_acc = train_model(train_loader, val_loader, test_loader, model, optimizer, scheduler,
                                                    device,
                                                    max_num_epochs)
                    vals.append(val_acc)
                    tests.append(test_acc)

            # Determine best model based on validation set performance.
            best_i = argmax(vals)
            best_test = tests[best_i]
            test_accuracies.append(best_test)

        if all_std:
            test_accuracies_complete.extend(test_accuracies)
        test_accuracies_all.append(np.array(test_accuracies).mean())

    if all_std:
        return (np.array(test_accuracies_all).mean(), np.array(test_accuracies_all).std(),
                np.array(test_accuracies_complete).std())
    else:
        return (np.array(test_accuracies_all).mean(), np.array(test_accuracies_all).std())
