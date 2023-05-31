import pandas as pd
import os
import numpy as np
import random
from create_datasets import create_train_dataset_supervised
from pycaret.classification import *

random.seed(42)

WORK_DIR = os.path.dirname(__file__)
FM_DIR = os.path.join(WORK_DIR, 'feature_maps')
BENCHMARK_DIR = os.path.join(WORK_DIR, 'benchmark')
DATA_DIR = os.path.join(WORK_DIR, 'data')
OUTPUTS_DIR = os.path.join(WORK_DIR, 'outputs')

if __name__ == '__main__':
    col_names = ['NTrSOrig', 'ATrSOrig', 'NTrSRes', 'ATrSRes', 'NTsSOrig', 'ATsSOrig', 'NTsSRes', 'ATsSRes',
                 'ubAccAvg', 'ubAccStd', 'ubAccMax', 'ubAccMin', 'ubAccMaxName', 'ubAccMinName',
                 'ubAUCAvg', 'ubAUCStd', 'ubAUCMax', 'ubAUCMin', 'ubAUCMaxName', 'ubAUCMinName',
                 'ubF1Avg', 'ubF1Std', 'ubF1Max', 'ubF1Min', 'ubF1MaxName', 'ubF1MinName',
                 'bAccAvg', 'bAccStd', 'bAccMax', 'bAccMin', 'bAccMaxName', 'bAccMinName',
                 'bAUCAvg', 'bAUCStd', 'bAUCMax', 'bAUCMin', 'bAUCMaxName', 'bAUCMinName',
                 'bF1Avg', 'bF1Std', 'bF1Max', 'bF1Min', 'bF1MaxName', 'bF1MinName'
                 ]
    sup_results_df = pd.DataFrame(columns=col_names)
    #for i in range(5):
    X_train_sup, y_train_sup, sizes_sup = create_train_dataset_supervised(1965)
    data_df = pd.DataFrame(X_train_sup)
    data_df['Labels'] = y_train_sup
    s = setup(data_df, target='Labels', session_id=42)
    sup_train_results = compare_models()
    print('Finnish')