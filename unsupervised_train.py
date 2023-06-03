import pandas as pd
import os
from os.path import join
import numpy as np
import random
from pyod.models.iforest import IForest
from pyod.models.ecod import ECOD
from pyod.utils.data import evaluate_print
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from create_datasets import create_train_dataset_unsupervised, create_test_dataset_balanced, create_test_dataset_unbalanced

random.seed(42)
WORK_DIR = os.path.dirname(__file__)
FM_DIR = os.path.join(WORK_DIR, 'feature_maps')
BENCHMARK_DIR = os.path.join(WORK_DIR, 'benchmark')
DATA_DIR = os.path.join(WORK_DIR, 'data')
OUTPUTS_DIR = os.path.join(WORK_DIR, 'outputs')


def test_unsupervised(model, dataset_id, dataset_type):
    if dataset_type == 'balanced':
        X_test, y_test, sizes = create_test_dataset_balanced(dataset_id)
    else:
        X_test, y_test, sizes = create_test_dataset_unbalanced(dataset_id)
    # get predictions on test data
    y_test_pred, confidence = model.predict(X_test, return_confidence=True)
    y_test_scores = model.decision_function(X_test)
    # evaluate and print the results
    print(f'Test results for balanced data: ')
    evaluate_print(model, y_test, y_test_scores)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    print('Accuracy: ' + str(accuracy_test))
    auc_test = roc_auc_score(y_test, y_test_scores)
    print('AUC: ' + str(auc_test))
    F1_score = f1_score(y_test, y_test_pred)
    print('F1: ' + str(F1_score))
    return accuracy_test, auc_test, F1_score, sizes

if __name__ == '__main__':
    col_names = ['NTrSOrig', 'NTsSOrig', 'ATsSOrig', 'NTsSRes', 'ATsSRes',
                 'ubAccAvg', 'ubAccStd', 'ubAccMax', 'ubAccMin', 'ubAccMaxName', 'ubAccMinName',
                 'ubAUCAvg', 'ubAUCStd', 'ubAUCMax', 'ubAUCMin', 'ubAUCMaxName', 'ubAUCMinName',
                 'ubF1Avg', 'ubF1Std', 'ubF1Max', 'ubF1Min', 'ubF1MaxName', 'ubF1MinName',
                 'bAccAvg', 'bAccStd', 'bAccMax', 'bAccMin', 'bAccMaxName', 'bAccMinName',
                 'bAUCAvg', 'bAUCStd', 'bAUCMax', 'bAUCMin', 'bAUCMaxName', 'bAUCMinName',
                 'bF1Avg', 'bF1Std', 'bF1Max', 'bF1Min', 'bF1MaxName', 'bF1MinName'
                 ]
    unsup_results_df = pd.DataFrame(columns=col_names)

    # model = ECOD(contamination=0.1)
    model = IForest(n_estimators=10, random_state=42)
    # train model
    X_train, y_train, sizes_train = create_train_dataset_unsupervised(0)
    data_df = pd.DataFrame(X_train)
    score = model.fit(X_train)
    acc, auc, f1, sizes = test_unsupervised(model, 0, 'balanced')
    print('Finnish')