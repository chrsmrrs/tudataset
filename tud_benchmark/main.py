from __future__ import division
from auxiliarymethods.evaluation import kernel_svm_evaluation, linear_svm_evaluation
from auxiliarymethods.evaluation import gnn_evaluation
from gnn_baselines.gin import GIN0

import kernel_baselines as kb
import auxiliarymethods.auxiliary_methods as aux
import auxiliarymethods.datasets as dp


def main():

    datataset = ["ENZYMES"]
    for d in datataset:

        dataset = d
        classes = dp.get_dataset(dataset)

        all_matrices = []
        for i in range(0, 6):
            gm = kb.compute_wloa_dense(dataset, i, True, False)
            gm_n = aux.normalize_gram_matrix(gm)
            all_matrices.append(gm)
            all_matrices.append(gm_n)
        print(kernel_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True))

        all_matrices = []
        for i in range(0, 6):
            gm = kb.compute_wl_1_dense(dataset, i, True, False)
            gm_n = aux.normalize_gram_matrix(gm)
            all_matrices.append(gm)
            all_matrices.append(gm_n)
        print(kernel_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True))



    # print(gnn_evaluation(GIN0, dataset, [2], [64], num_repetitions=10, all_std=True))


if __name__ == "__main__":
    main()
