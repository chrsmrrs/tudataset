from __future__ import division
from auxiliarymethods.evaluation import gnn_evaluation
from gnn_baselines.gnn_architectures import GIN0, GINWithJK
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, SGDRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mse
import kernel_baselines as kb
import auxiliarymethods.auxiliary_methods as aux
import auxiliarymethods.datasets as dp
from sklearn.preprocessing import StandardScaler
import numpy as np


def main():
    results = gnn_evaluation(GINWithJK, "PROTEINS", [1,2,3,4,5], [64], max_num_epochs=100, batch_size=25, start_lr=0.001, num_repetitions=10,
                   all_std=True)
    print(results)

    results = gnn_evaluation(GIN0, "MUTAG", [1,2,3,4,5], [16, 32, 64, 128], max_num_epochs=200, batch_size=25, start_lr=0.001, num_repetitions=10,
                   all_std=True)

    print(results)


if __name__ == "__main__":
    main()
