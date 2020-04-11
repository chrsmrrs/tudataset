from __future__ import division

import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from sklearn.linear_model import SGDRegressor, Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error as mse

# Return arg max of iterable, e.g., a list.
def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]




# 10-CV for linear svm with sparse feature vectors and hyperparameter selection.
def sgd_regressor_evaluation(all_feature_matrices, targets, train_index, val_index, test_index, num_repetitions=5,
                             alpha=[0.00001, 0.0001, 0.001, 0.01]):
    # Acc. over all repetitions.
    test_accuracies = []

    for _ in range(num_repetitions):

        val_accuracies = []
        models = []
        for f in all_feature_matrices:

            train = f[train_index]
            val = f[val_index]

            c_train = targets[train_index]
            c_val = targets[val_index]

            for a in alpha:
                clf = SGDRegressor(alpha=a)
                clf.fit(train, c_train)
                p = clf.predict(val)
                r = mse(c_val, p)

                models.append(clf)
                val_accuracies.append(r)

        best_i = argmax(val_accuracies)
        best_model = models[best_i]

        # Eval. model on test split that performed best on val. split.
        test = all_feature_matrices[int(best_i / len(alpha))][test_index]
        c_test = targets[test_index]
        p = best_model.predict(test)
        a = mse(c_test, p)
        test_accuracies.append(a)

    return (np.array(test_accuracies).mean(), np.array(test_accuracies).std())


def ridge_regressor_evaluation(all_feature_matrices, targets, train_index, val_index, test_index, num_repetitions=5,
                             alpha=[0.01, 0.1, 1.0, 2.0]):
    # Acc. over all repetitions.
    test_accuracies = []

    for _ in range(num_repetitions):

        val_accuracies = []
        models = []
        for f in all_feature_matrices:

            train = f[train_index]
            val = f[val_index]

            c_train = targets[train_index]
            c_val = targets[val_index]

            for a in alpha:
                clf = Ridge(alpha=a)
                clf.fit(train, c_train)
                p = clf.predict(val)
                r = mse(c_val, p)

                models.append(clf)
                val_accuracies.append(r)

        best_i = argmax(val_accuracies)
        best_model = models[best_i]

        # Eval. model on test split that performed best on val. split.
        test = all_feature_matrices[int(best_i / len(alpha))][test_index]
        c_test = targets[test_index]
        p = best_model.predict(test)
        a = mse(c_test, p)
        test_accuracies.append(a)

    return (np.array(test_accuracies).mean(), np.array(test_accuracies).std())


def kernel_ridge_regressor_evaluation(all_feature_matrices, targets, train_index, val_index, test_index, num_repetitions=5,
                             alpha=[0.01, 0.1, 1.0, 2.0]):
    # Acc. over all repetitions.
    test_accuracies = []

    for _ in range(num_repetitions):

        val_accuracies = []
        models = []
        for f in all_feature_matrices:

            train = f[train_index]
            train = train[:,train_index]
            val = f[val_index]
            val = val[:,train_index]

            c_train = targets[train_index]
            c_val = targets[val_index]

            for a in alpha:
                clf = KernelRidge(alpha=a, kernel="precomputed")
                clf.fit(train, c_train)
                p = clf.predict(val)
                r = mse(c_val, p)

                models.append(clf)
                val_accuracies.append(r)

            best_i = argmax(val_accuracies)
            best_model = models[best_i]

            # Eval. model on test split that performed best on val. split.
            test = all_feature_matrices[int(best_i / len(alpha))][test_index]
            test = test[:, train_index]
            c_test = targets[test_index]

            p = best_model.predict(test)
            a = mse(c_test, p)
            test_accuracies.append(a)

        return (np.array(test_accuracies).mean(), np.array(test_accuracies).std())

# 10-CV for linear svm with sparse feature vectors and hyperparameter selection.
def linear_svm_evaluation(all_feature_matrices, classes, num_repetitions=10,
                          C=[10 ** 3, 10 ** 2, 10 ** 1, 10 ** 0, 10 ** -1, 10 ** -2, 10 ** -3], all_std=False,
                          primal=True):
    # Acc. over all repetitions.
    test_accuracies_all = []
    # All acc. over all folds and repetitions.
    test_accuracies_complete = []

    for i in range(num_repetitions):
        # Test acc. over all folds.
        test_accuracies = []
        kf = KFold(n_splits=10, shuffle=True)

        for train_index, test_index in kf.split(list(range(len(classes)))):
            val_accuracies = []
            models = []
            for f in all_feature_matrices:
                train_index, val_index = train_test_split(train_index, test_size=0.1)
                train = f[train_index]
                val = f[val_index]

                c_train = classes[train_index]
                c_val = classes[val_index]

                for c in C:
                    clf = LinearSVC(C=c, dual=not primal)
                    clf.fit(train, c_train)
                    p = clf.predict(val)
                    a = np.sum(np.equal(p, c_val)) / val.shape[0]

                    models.append(clf)
                    val_accuracies.append(a)

            best_i = argmax(val_accuracies)
            best_model = models[best_i]

            # Eval. model on test split that performed best on val. split.
            test = all_feature_matrices[int(best_i / len(C))][test_index]
            c_test = classes[test_index]
            p = best_model.predict(test)
            a = np.sum(np.equal(p, c_test)) / test.shape[0]
            test_accuracies.append(a * 100.0)

            if all_std:
                test_accuracies_complete.append(a * 100.0)

        test_accuracies_all.append(float(np.array(test_accuracies).mean()))

    if all_std:
        return (np.array(test_accuracies_all).mean(), np.array(test_accuracies_all).std(),
                np.array(test_accuracies_complete).std())
    else:
        return (np.array(test_accuracies_all).mean(), np.array(test_accuracies_all).std())


# 10-CV for kernel svm and hyperparameter selection.
def kernel_svm_evaluation(all_matrices, classes, num_repetitions=10,
                          C=[10 ** 3, 10 ** 2, 10 ** 1, 10 ** 0, 10 ** -1, 10 ** -2, 10 ** -3], all_std=False):
    test_accuracies_all = []
    test_accuracies_complete = []

    for i in range(num_repetitions):
        # Test acc. over all folds.
        test_accuracies = []
        kf = KFold(n_splits=10, shuffle=True)

        for train_index, test_index in kf.split(list(range(len(classes)))):
            train_index, val_index = train_test_split(train_index, test_size=0.1)
            val_accuracies = []
            models = []
            for gram_matrix in all_matrices:
                train = gram_matrix[train_index, :]
                train = train[:, train_index]
                val = gram_matrix[val_index, :]
                val = val[:, train_index]

                c_train = classes[train_index]
                c_val = classes[val_index]

                for c in C:
                    clf = SVC(C=c, kernel="precomputed")
                    clf.fit(train, c_train)
                    p = clf.predict(val)
                    a = np.sum(np.equal(p, c_val)) / val.shape[0]

                    models.append(clf)
                    val_accuracies.append(a)

            best_i = argmax(val_accuracies)
            best_model = models[best_i]

            test = all_matrices[int(best_i / len(C))][test_index, :]
            test = test[:, train_index]
            c_test = classes[test_index]
            p = best_model.predict(test)
            a = np.sum(np.equal(p, c_test)) / test.shape[0]
            test_accuracies.append(a * 100.0)

            if all_std:
                test_accuracies_complete.append(a * 100.0)

        test_accuracies_all.append(float(np.array(test_accuracies).mean()))

    if all_std:
        return (np.array(test_accuracies_all).mean(), np.array(test_accuracies_all).std(),
                np.array(test_accuracies_complete).std())
    else:
        return (np.array(test_accuracies_all).mean(), np.array(test_accuracies_all).std())


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


# Train GNN modell
def train_model(train_loader, val_loader, test_loader, model, optimizer, device, num_epochs):
    test_acc = None
    for epoch in range(1, num_epochs):
        train(train_loader, model, optimizer, device)

    test_acc = test(test_loader, model, device)
    val_acc = test(val_loader, model, device)
    return val_acc, test_acc


# 10-CV for GNN training and hyperparameter selection.
def gnn_evaluation(gnn, ds_name, layers, hidden, num_epochs=100, batch_size=25, lr=0.1, num_repetitions=10,
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

        # TODO rest????############################################################

        for train_index, test_index in kf.split(list(range(len(dataset)))):
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
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    val_acc, test_acc = train_model(train_loader, val_loader, test_loader, model, optimizer, device,
                                                    num_epochs)
                    vals.append(val_acc)
                    tests.append(test_acc)

            # Determine best model.
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
