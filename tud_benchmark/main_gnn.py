from __future__ import division
from auxiliarymethods.evaluation import gnn_evaluation
from gnn_baselines.gnn_architectures import GINWithJK, GraphSAGEWithJK
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, SGDRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mse
import kernel_baselines as kb
import auxiliarymethods.auxiliary_methods as aux
import auxiliarymethods.datasets as dp
from sklearn.preprocessing import StandardScaler
import numpy as np

# TODO: Add one-hot.

def main():
    datasets = [
        ["ENZYMES", True],
        # ["IMDB-BINARY", False], ["IMDB-MULTI", False],["REDDIT-BINARY", False]
        ["NCI1", True],
        ["PROTEINS", True]]

    for d, use_labels in datasets:
        classes = dp.get_dataset(d)

        results = gnn_evaluation(GINWithJK, d, [1, 2, 3, 4, 5], [16, 32, 64, 128], max_num_epochs=200, batch_size=32, start_lr=0.01,
                                 num_repetitions=10,
                                 all_std=True)
        print(results)

        results = gnn_evaluation(GraphSAGEWithJK, d, [1, 2, 3, 4, 5], [16, 32, 64, 128], max_num_epochs=200,
                                 batch_size=32, start_lr=0.01, num_repetitions=10,
                                 all_std=True)
        print(results)

    print(results)


if __name__ == "__main__":
    main()
