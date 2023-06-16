import pandas as pd
import os
from os.path import join
import numpy as np
import random
from pyod.models.iforest import IForest
from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD
from pyod.models.knn import KNN
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
    # since in our implementation anomaly is represented as 0 and in PYOD predict anomaly is 1,
    # change y_test accordingly
    y_test = 1 - y_test
    # get predictions on test data
    y_test_pred, confidence = model.predict(X_test, return_confidence=True)
    y_test_scores = model.decision_function(X_test)
    # evaluate and print the results
    evaluate_print(model, y_test, y_test_scores)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    print('Accuracy: ' + str(accuracy_test))
    auc_test = roc_auc_score(y_test, y_test_scores)
    print('AUC: ' + str(auc_test))
    F1_score = f1_score(y_test, y_test_pred)
    print('F1: ' + str(F1_score))
    return accuracy_test, auc_test, F1_score, sizes


def update_results(df, cols_list):
    result = []
    for col in cols_list:
        avg = df[col].mean()
        std = df[col].std()
        max = df[col].max()
        min = df[col].min()
        max_name_idx = df[col].idxmax()
        max_model_name = df.loc[max_name_idx, 'ModelName']
        min_name_idx = df[col].idxmin()
        min_model_name = df.loc[min_name_idx, 'ModelName']
        result.append([avg, std, max, min, max_model_name, min_model_name])
    result.append(df.iloc[0, [0, 1, 2, 3, 4, 5]])
    flat_list = [item for sublist in result for item in sublist]
    result_df = pd.DataFrame([flat_list])
    return result_df


if __name__ == '__main__':
    result_col_names = ['bAccAvg', 'bAccStd', 'bAccMax', 'bAccMin', 'bAccMaxName', 'bAccMinName',
                 'bAUCAvg', 'bAUCStd', 'bAUCMax', 'bAUCMin', 'bAUCMaxName', 'bAUCMinName',
                 'bF1Avg', 'bF1Std', 'bF1Max', 'bF1Min', 'bF1MaxName', 'bF1MinName',
                 'ubAccAvg', 'ubAccStd', 'ubAccMax', 'ubAccMin', 'ubAccMaxName', 'ubAccMinName',
                 'ubAUCAvg', 'ubAUCStd', 'ubAUCMax', 'ubAUCMin', 'ubAUCMaxName', 'ubAUCMinName',
                 'ubF1Avg', 'ubF1Std', 'ubF1Max', 'ubF1Min', 'ubF1MaxName', 'ubF1MinName',
                 'NTrSOrig',  'ATrSOrig','NTsSBalanced', 'ATsSBalanced', 'NTsSUnBalanced', 'ATsSUnBalanced',
                 ]
    unsup_results_df = pd.DataFrame()
    # Models to test
    model_name_list = ['IForest', 'ECOD', 'COPOD', 'KNN']
    model_list = [IForest(n_estimators=10, random_state=42),
                 ECOD(), COPOD(), KNN()]

    #############################
    # Run on datasets
    for dataset_id in range(2):
        results = []
        for model in model_list:
            # train model
            X_train, y_train, sizes_train = create_train_dataset_unsupervised(dataset_id, include_anomaly=False)
            data_df = pd.DataFrame(X_train)
            # cont = sizes_train[1]/(sizes_train[0]+ sizes_train[1])
            cont = 0.1
            # update contamination
            model.set_params(contamination=cont)
            score = model.fit(X_train)
            acc_b, auc_b, f1_b, sizes_b = test_unsupervised(model, dataset_id, 'balanced')
            acc_ub, auc_ub, f1_ub, sizes_ub = test_unsupervised(model, dataset_id, 'unbalanced')
            results.append([sizes_train[0], sizes_train[1], sizes_b[0], sizes_b[1], sizes_ub[0], sizes_ub[1],
                       acc_b, auc_b, f1_b, acc_ub, auc_ub, f1_ub])
        results_matrix = np.asarray(results)
        col_names = ['NTrSOrig', 'ATrSOrig', 'NTsSBalanced', 'ATsSBalanced', 'NTsSUnBalanced', 'ATsSUnBalanced',
                     'bAcc', 'bAUC', 'bF1',
                     'ubAcc', 'ubAUC', 'ubF1',
                     ]
        unsup_ds_results_df = pd.DataFrame(results_matrix, columns=col_names)
        unsup_ds_results_df['ModelName'] = model_name_list
        # path = join(OUTPUTS_DIR, 'unsupervised_ds_results.csv')
        # unsup_ds_results_df.to_csv(path)
        ds_result_df = update_results(unsup_ds_results_df, ['bAcc', 'bAUC', 'bF1', 'ubAcc', 'ubAUC', 'ubF1'])
        unsup_results_df = pd.concat([unsup_results_df, ds_result_df], axis=0)
    #############################
    # save results to csv file
    unsup_results_df.columns = result_col_names
    unsup_results_df = unsup_results_df.reset_index(drop=True)
    path = join(OUTPUTS_DIR, 'unsupervised_results.csv')
    unsup_results_df.to_csv(path, index=True)
    print('Finnish')