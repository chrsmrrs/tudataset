import auxiliarymethods.auxiliary_methods as aux
import auxiliarymethods.datasets as dp
import kernel_baselines as kb
from auxiliarymethods.kernel_evaluation import linear_svm_evaluation
from auxiliarymethods.kernel_evaluation import kernel_svm_evaluation
import numpy as np

import pandas as pd
import auxiliarymethods.datasets as dp
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVC
from scipy import sparse as sp


def main():
    path = "/home/morris/localwl_dev/svm/SVM/src/EXPSPARSE/"

    # for name in ["Yeast", "YeastH", "UACC257", "UACC257H", "OVCAR-8", "OVCAR-8H"]:
    #     for algorithm in ["WL", "LWL2", "LWLP2"]:

    for name in ["Yeast"]:
        for algorithm in ["LWLP2"]:

            print(name)
            print(algorithm)

            # Collect feature matrices over all iterations
            all_feature_matrices = []

            for i in range(0, 3):
                # Load feature matrices.
                feature_vector = pd.read_csv(path + name + "__" + algorithm + "_" + str(i), header=1,
                                             delimiter=" ").to_numpy()

                feature_vector = feature_vector.astype(int)
                feature_vector[:, 0] = feature_vector[:, 0] - 1
                feature_vector[:, 1] = feature_vector[:, 1] - 1
                feature_vector[:, 2] = feature_vector[:, 2] + 1

                xmax = int(feature_vector[:, 0].max())
                ymax = int(feature_vector[:, 1].max())

                feature_vector = sp.coo_matrix((feature_vector[:, 2], (feature_vector[:, 0], feature_vector[:, 1])),
                                               shape=(xmax + 1, ymax + 1))
                feature_vector = feature_vector.tocsr()
                # TODO: Apply normalization.
                all_feature_matrices.append(feature_vector)
            print("### Data loading done.")

    print(len(all_feature_matrices))
    classes = dp.get_dataset("Yeast")
    acc, s_1, s_2 = linear_svm_evaluation(all_feature_matrices, classes, num_repetitions=1, all_std=True)
    print("yeast" + " " + "LWLP2SP " + str(acc) + " " + str(s_1) + " " + str(s_2))



if __name__ == "__main__":
    main()
