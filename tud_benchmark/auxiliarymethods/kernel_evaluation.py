import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score

# 10-CV for linear svm with sparse feature vectors and hyperparameter selection.
def linear_svm_evaluation(all_feature_matrices, classes, num_repetitions=10,
                          C=[10 ** 3, 10 ** 2, 10 ** 1, 10 ** 0, 10 ** -1, 10 ** -2, 10 ** -3], all_std=False, primal=True, max_iterations=-1):
    # Acc. over all repetitions.
    test_accuracies_all = []
    # All acc. over all folds and repetitions.
    test_accuracies_complete = []

    for i in range(num_repetitions):
        # Test acc. over all folds.
        test_accuracies = []
        kf = KFold(n_splits=10, shuffle=True)

        for train_index, test_index in kf.split(list(range(len(classes)))):
            # Sample 10% for validation.
            train_index, val_index = train_test_split(train_index, test_size=0.1)
            best_val_acc = 0.0
            best_test = 0.0

            for feature_vector in all_feature_matrices:
                train = feature_vector[train_index, :]
                val = feature_vector[val_index, :]
                test = feature_vector[test_index, :]


                c_train = classes[train_index]
                c_val = classes[val_index]
                c_test = classes[test_index]

                for c in C:
                    #clf = LinearSVC(C=c, dual=not primal, max_iter=max_iterations)
                    clf = LinearSVC(C=c)
                    clf.fit(train, c_train)
                    val_acc = accuracy_score(c_val, clf.predict(val)) * 100.0

                    if val_acc > best_val_acc:
                        # Get test acc.
                        best_val_acc = val_acc
                        best_test = accuracy_score(c_test, clf.predict(test)) * 100.0

            test_accuracies.append(best_test)
            if all_std:
                test_accuracies_complete.append(best_test)


        test_accuracies_all.append(float(np.array(test_accuracies).mean()))

    if all_std:
        return (np.array(test_accuracies_all).mean(), np.array(test_accuracies_all).std(),
                np.array(test_accuracies_complete).std())
    else:
        return (np.array(test_accuracies_all).mean(), np.array(test_accuracies_all).std())


# 10-CV for kernel svm and hyperparameter selection.
def kernel_svm_evaluation(all_matrices, classes, num_repetitions=10, C=[10 ** 3, 10 ** 2, 10 ** 1, 10 ** 0, 10 ** -1, 10 ** -2, 10 ** -3], all_std=False, max_iterations=-1):
    # Acc. over all repetitions.
    test_accuracies_all = []
    # All acc. over all folds and repetitions.
    test_accuracies_complete = []


    for i in range(num_repetitions):
        print(i)
        # Test acc. over all folds.
        test_accuracies = []
        kf = KFold(n_splits=10, shuffle=True)

        for train_index, test_index in kf.split(list(range(len(classes)))):
            # Sample 10% for validation.
            train_index, val_index = train_test_split(train_index, test_size=0.1)
            best_val_acc = 0.0
            best_test = 0.0

            for gram_matrix in all_matrices:
                train = gram_matrix[train_index, :]
                train = train[:, train_index]
                val = gram_matrix[val_index, :]
                val = val[:, train_index]
                test = gram_matrix[test_index, :]
                test = test[:, train_index]

                c_train = classes[train_index]
                c_val = classes[val_index]
                c_test = classes[test_index]

                for c in C:
                    clf = SVC(C=c, kernel="precomputed", tol=0.001, max_iter=max_iterations)
                    clf.fit(train, c_train)
                    val_acc = accuracy_score(c_val, clf.predict(val)) * 100.0

                    if val_acc > best_val_acc:
                        # Get test acc.
                        best_val_acc = val_acc
                        best_test = accuracy_score(c_test, clf.predict(test)) * 100.0

            test_accuracies.append(best_test)
            if all_std:
                test_accuracies_complete.append(best_test)

        test_accuracies_all.append(float(np.array(test_accuracies).mean()))

    if all_std:
        return (np.array(test_accuracies_all).mean(), np.array(test_accuracies_all).std(),
                np.array(test_accuracies_complete).std())
    else:
        return (np.array(test_accuracies_all).mean(), np.array(test_accuracies_all).std())
