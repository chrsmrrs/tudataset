from __future__ import division
from auxiliarymethods.evaluation import kernel_svm_evaluation, linear_svm_evaluation, sgd_regressor_evaluation, \
    ridge_regressor_evaluation, kernel_ridge_regressor_evaluation
from auxiliarymethods.evaluation import gnn_evaluation
from gnn_baselines.gin import GIN0
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, SGDRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mse
import kernel_baselines as kb
import auxiliarymethods.auxiliary_methods as aux
import auxiliarymethods.datasets as dp
from sklearn.preprocessing import StandardScaler
import numpy as np


def main():




if __name__ == "__main__":
    main()
