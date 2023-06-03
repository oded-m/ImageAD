import pandas as pd
import os
from os.path import join
import numpy as np
import random
from create_datasets import create_train_dataset_supervised, create_test_dataset_balanced, create_test_dataset_unbalanced
from pycaret.classification import *
from pycaret.datasets import get_data

random.seed(42)

WORK_DIR = os.path.dirname(__file__)
FM_DIR = os.path.join(WORK_DIR, 'feature_maps')
BENCHMARK_DIR = os.path.join(WORK_DIR, 'benchmark')
DATA_DIR = os.path.join(WORK_DIR, 'data')
OUTPUTS_DIR = os.path.join(WORK_DIR, 'outputs')


if __name__ == '__main__':
    col_names = ['bAccAvg', 'bAccStd', 'bAccMax', 'bAccMin', 'bAccMaxName', 'bAccMinName',
                 'bAUCAvg', 'bAUCStd', 'bAUCMax', 'bAUCMin', 'bAUCMaxName', 'bAUCMinName',
                 'bF1Avg', 'bF1Std', 'bF1Max', 'bF1Min', 'bF1MaxName', 'bF1MinName',
                 'ubAccAvg', 'ubAccStd', 'ubAccMax', 'ubAccMin', 'ubAccMaxName', 'ubAccMinName',
                 'ubAUCAvg', 'ubAUCStd', 'ubAUCMax', 'ubAUCMin', 'ubAUCMaxName', 'ubAUCMinName',
                 'ubF1Avg', 'ubF1Std', 'ubF1Max', 'ubF1Min', 'ubF1MaxName', 'ubF1MinName',
                 'NTrSOrig', 'ATrSOrig', 'NTrSBalanced', 'ATrSBalanced', 'NTsSOrig', 'ATsSOrig',
                 'NTsSUnBalanced', 'ATsSUnBalanced'
                 ]
    sup_results_df = pd.DataFrame(columns=col_names)

    X_train_sup, y_train_sup, sizes_sup = create_train_dataset_supervised(0)
    data_df = pd.DataFrame(X_train_sup)
    data_df['Labels'] = y_train_sup
    # test balanced dataset
    X_test_b, y_test_b, sizes_b = create_test_dataset_balanced(0)
    test_b_df = pd.DataFrame(X_test_b)
    test_b_df['Labels'] = y_test_b
    sb = setup(data_df, target='Labels', session_id=42, n_jobs=1, test_data=test_b_df, index=False)
    best = compare_models(include=['lr', 'knn', 'dummy'], fold=5)
    sup_train_results = pull()
    sup_train_results.to_csv(join(OUTPUTS_DIR, 'results_balanced_cv.csv'))
    ###########
    # to test need to use predict
    ###########
    # test unbalanced dataset
    X_test_u, y_test_u, sizes_u = create_test_dataset_unbalanced(0)
    test_u_df = pd.DataFrame(X_test_u)
    test_u_df['Labels'] = y_test_u
    su = setup(data_df, target='Labels', session_id=42, n_jobs=1, test_data=test_u_df, index=False)
    best = compare_models(include=['lr', 'knn', 'dummy'], fold=5)
    sup_train_results_u = pull()
    sup_train_results_u.to_csv(join(OUTPUTS_DIR, 'results_unbalanced_cv.csv'))
    ###########
    # to test need to use predict
    ###########
    print('Finnish')