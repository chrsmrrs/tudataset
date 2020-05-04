import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC


# Return arg max of iterable, e.g., a list.
def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]


# 10-CV for linear svm with sparse feature vectors and hyperparameter selection.
def linear_svm_evaluation(all_feature_matrices, classes, num_repetitions=10,
                          C=[10 ** 3, 10 ** 2, 10 ** 1, 10 ** 0, 10 ** -1, 10 ** -2, 10 ** -3], all_std=False,
                          primal=True, max_iterations=-1):
    # Acc. over all repetitions.
    test_accuracies_all = []
    # All acc. over all folds and repetitions.
    test_accuracies_complete = []

    for i in range(num_repetitions):
        # Test acc. over all folds.
        test_accuracies = []
        kf = KFold(n_splits=10, shuffle=True)

        for train_index, test_index in kf.split(list(range(len(classes)))):
            best_val_acc = 0.0
            best_test = 0.0

            models = []
            for f in all_feature_matrices:
                # Sample 10% for validation.
                train_index, val_index = train_test_split(train_index, test_size=0.1)
                train = f[train_index]
                val = f[val_index]
                test = f[train_index]
                c_train = classes[train_index]
                c_val = classes[val_index]
                c_test = classes[test_index]

                for c in C:
                    # Default values of https://github.com/cjlin1/liblinear/blob/master/README.
                    if not primal:
                        clf = LinearSVC(C=c, dual=not primal, max_iter=max_iterations, tol=0.1, penalty="l2",
                                        loss="hinge")
                    else:
                        clf = LinearSVC(C=c, dual=not primal, max_iter=max_iterations, tol=0.01, penalty="l2",
                                        loss="hinge")

                    clf.fit(train, c_train)
                    p = clf.predict(val)
                    val_acc = np.sum(np.equal(p, c_val)) / val.shape[0]

                    if val_acc < best_val_acc:
                        best_val_acc = val_acc

                    c_test = classes[test_index]
                    p = clf.predict(test)
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
                    clf = SVC(C=c, kernel="precomputed", tol=0.001)
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
