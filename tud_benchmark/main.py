from __future__ import division
from auxiliarymethods.evaluation import kernel_svm_evaluation, linear_svm_evaluation
from auxiliarymethods.evaluation import gnn_evaluation
from gnn_baselines.gin import GIN0
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, SGDRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mse
import kernel_baselines as kb
import auxiliarymethods.auxiliary_methods as aux
import auxiliarymethods.datasets as dp
from sklearn.preprocessing import StandardScaler


def main():
    dataset = "deezer_ego_nets"
    classes = dp.get_dataset(dataset)
    print(classes.sum()/len(classes))

    print("WL")
    all_matrices = []
    for i in range(1, 4):
        print(i)
        gm = kb.compute_wl_1_sparse(dataset, i, False, False)
        gm_n = aux.normalize_feature_vector(gm)
        all_matrices.append(gm)
        all_matrices.append(gm_n)
    print("###")
    print(linear_svm_evaluation(all_matrices, classes, num_repetitions=1, all_std=True, primal=True))

    print("WL")
    all_matrices = []
    for i in range(3, 4):
        print(i)
        gm = kb.compute_lwlp_2_dense(dataset, i, False, False)
        gm_n = aux.normalize_gram_matrix(gm)
        #all_matrices.append(gm)
        all_matrices.append(gm_n)
    print("###")
    print(kernel_svm_evaluation(all_matrices, classes, num_repetitions=1, all_std=True))


    exit()


    dp.get_dataset("ZINC_train", regression=True)
    dp.get_dataset("ZINC_val", regression=True)
    dp.get_dataset("ZINC_test", regression=True)

    indices_train = []
    indices_val = []
    indices_test = []

    infile = open("datasets/test.index.txt", "r")
    for line in infile:
        indices_test = line.split(",")
        indices_test = [int(i) for i in indices_test]

    infile = open("datasets/val.index.txt", "r")
    for line in infile:
        indices_val = line.split(",")
        indices_val = [int(i) for i in indices_val]

    infile = open("datasets/train.index.txt", "r")
    for line in infile:
        indices_train = line.split(",")
        indices_train = [int(i) for i in indices_train]

    print("###")

    targets_train = kb.read_targets("ZINC_train", indices_train)
    targets_val = kb.read_targets("ZINC_val", indices_val)
    targets_test = kb.read_targets("ZINC_test", indices_test)

    print("###")

    f = kb.compute_wl_1_sparse_ZINC(4, True, True, indices_train, indices_val, indices_test)
    print("###")

    print(f.shape)

    X_train = f[0:10000]
    X_test = f[11000:12000]



    reg = SGDRegressor(max_iter=1000).fit(X_train, targets_train)
    p = reg.predict(X_test)
    print(mse(targets_test, p))
    exit()

    datataset = [["ENZYMES", True],
                 ["IMDB-BINARY", False], ["IMDB-MULTI", False],
                 ["NCI1", True],
                 ["PROTEINS", True],
                 ["REDDIT-BINARY", False],
                 ["deezer_ego_nets", False]]
    for d, use_labels in datataset:

        dataset = d
        classes = dp.get_dataset(dataset)

        print("WL1")
        all_matrices = []
        for i in range(1, 6):
            gm = kb.compute_wl_1_dense(dataset, i, use_labels, False)
            gm_n = aux.normalize_gram_matrix(gm)
            all_matrices.append(gm_n)
        print("###")
        print(kernel_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True))

        print("WLOA")
        all_matrices = []
        for i in range(1, 6):
            gm = kb.compute_wloa_dense(dataset, i, use_labels, False)
            gm_n = aux.normalize_gram_matrix(gm)
            all_matrices.append(gm_n)
        print("###")
        print(kernel_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True))

        print("GR")
        all_matrices = []
        gm = kb.compute_graphlet_dense(dataset, use_labels, False)
        gm_n = aux.normalize_gram_matrix(gm)
        all_matrices.append(gm_n)
        print("###")
        print(kernel_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True))

        print("SP")
        all_matrices = []
        gm = kb.compute_shortestpath_dense(dataset, use_labels)
        gm_n = aux.normalize_gram_matrix(gm)
        all_matrices.append(gm_n)
        print("###")
        print(kernel_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True))

        print("LWL2")
        all_matrices = []
        for i in range(1, 6):
            gm = kb.compute_lwl_2_dense(dataset, i, use_labels, False)
            gm_n = aux.normalize_gram_matrix(gm)
            all_matrices.append(gm_n)
        print("###")
        print(kernel_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True))

        print("LWLP2")
        all_matrices = []
        for i in range(1, 6):
            gm = kb.compute_lwlp_2_dense(dataset, i, use_labels, False)
            gm_n = aux.normalize_gram_matrix(gm)
            all_matrices.append(gm_n)
        print("###")
        print(kernel_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True))

        print("LWL2OA")
        all_matrices = []
        for i in range(1, 6):
            gm = kb.compute_lwl_2_wloa_dense(dataset, i, use_labels, False)
            gm_n = aux.normalize_gram_matrix(gm)
            all_matrices.append(gm_n)
        print("###")
        print(kernel_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True))

        print("LWLP2OA")
        all_matrices = []
        for i in range(1, 6):
            gm = kb.compute_lwlp_2_wloa_dense(dataset, i, use_labels, False)
            gm_n = aux.normalize_gram_matrix(gm)
            all_matrices.append(gm_n)
        print("###")
        print(kernel_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True))


if __name__ == "__main__":
    main()
