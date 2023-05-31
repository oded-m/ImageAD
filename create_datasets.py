import pandas as pd
import os
import pickle
import numpy as np
from os.path import join
from imblearn.over_sampling import SMOTE
import random

random.seed(42)

WORK_DIR = os.path.dirname(__file__)
FM_DIR = os.path.join(WORK_DIR, 'feature_maps')
BENCHMARK_DIR = os.path.join(WORK_DIR, 'benchmark')
DATA_DIR = os.path.join(WORK_DIR, 'data')
OUTPUTS_DIR = os.path.join(WORK_DIR, 'outputs')


def get_dataset_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    encodings_arr_all = np.array([], dtype=np.int64).reshape(0, 512)
    labels_arr_all = np.array([], dtype=np.int64).reshape(0, 1)
    fm_list = df['FeatureMapFile'].unique().tolist()
    for fm in fm_list:
        fmi_list = df[df['FeatureMapFile'] == fm]['FeatureMapIndex'].tolist()
        encoding_f = os.path.join(FM_DIR, fm)
        with open(encoding_f, 'rb') as f:
            fm_df = pickle.load(f)
        encodings_arr = fm_df.loc[fmi_list, 'FeatureMap'].values
        encodings_arr = np.stack(encodings_arr).squeeze()
        encodings_arr_all = np.concatenate((encodings_arr_all, encodings_arr))
        labels_arr = df[df['FeatureMapFile'] == fm]['IsNormal'].values
        labels_arr = labels_arr.squeeze().reshape(labels_arr.shape[0], 1)
        labels_arr_all = np.concatenate((labels_arr_all, labels_arr))
    return encodings_arr_all, labels_arr_all


def balance_dataset(normal_encodings, normal_labels, anomaly_encodings, anomaly_labels):
    normal_size = normal_encodings.shape[0]
    anomaly_size = anomaly_encodings.shape[0]
    if normal_size > anomaly_size:
        ratio = normal_size / anomaly_size
        large_encodings = normal_encodings
        large_labels = normal_labels
        large_size = normal_size
        small_encodings = anomaly_encodings
        small_labels = anomaly_labels
        small_size = anomaly_size
    else:
        ratio = anomaly_size / normal_size
        large_encodings = anomaly_encodings
        large_labels = anomaly_labels
        large_size = anomaly_size
        small_encodings = normal_encodings
        small_labels = normal_labels
        small_size = normal_size
    X_train = np.concatenate((large_encodings, small_encodings))
    y_train = np.concatenate((large_labels, small_labels))

    if (small_size < 800) & (small_size > 100):
        # downsample larger
        idx = random.sample(range(large_size), small_size)
        large_encodings1 = large_encodings[idx]
        large_labels1 = large_labels[idx]
        X_res = np.concatenate((large_encodings1, small_encodings))
        y_res = np.concatenate((large_labels1, small_labels))
    elif small_size < 100:
        # don't use dataset
        X_res = np.nan
        y_res = np.nan
    else:
        if ratio <= 1.3:
            # upsample smaller to large size
            sm = SMOTE(sampling_strategy=1, random_state=42)
            X_res, y_res = sm.fit_resample(X_train, y_train)
        elif ratio > 1.3:
            # downsample larger to 1.3*smaller
            idx = random.sample(range(large_size), int(1.3 * small_size))
            large_encodings1 = large_encodings[idx]
            large_labels1 = large_labels[idx]
            X_train = np.concatenate((large_encodings1, small_encodings))
            y_train = np.concatenate((large_labels1, small_labels))
            # upsample smaller to 1.3*smaller
            sm = SMOTE(sampling_strategy=1, random_state=42)
            X_res, y_res = sm.fit_resample(X_train, y_train)
    return X_res, y_res


def create_train_dataset_supervised(dataset_id):
    train_f = 'TRAIN_DATASET_' + str(dataset_id) + '.csv'
    train_path = join(BENCHMARK_DIR, train_f)
    sup_train_f = 'SUP_TRAIN_DATASET_' + str(dataset_id) + '.csv'
    sup_train_path = join(BENCHMARK_DIR, sup_train_f)
    normal_encodings, normal_labels = get_dataset_from_csv(train_path)
    anomaly_encodings, anomaly_labels = get_dataset_from_csv(sup_train_path)
    X_train, y_train = balance_dataset(normal_encodings, normal_labels, anomaly_encodings, anomaly_labels)
    n_normal_orig = normal_labels.shape[0]
    n_anomaly_orig = anomaly_labels.shape[0]
    n_normal_res = y_train.sum()
    n_anomaly_res = y_train.shape[0] - n_normal_res
    sizes = (n_normal_orig, n_normal_res, n_anomaly_orig, n_anomaly_res)
    return X_train, y_train, sizes


def create_test_dataset_balanced(dataset_id):
    test_f = 'TEST_DATASET_' + str(dataset_id) + '.csv'
    test_path = join(BENCHMARK_DIR, test_f)
    X_test_unbalanced, y_test_unbalanced = get_dataset_from_csv(test_path)
    df = pd.DataFrame(X_test_unbalanced)
    df['Labels'] = y_test_unbalanced
    # divide df to normal and anomaly
    normal_encodings = df[df['Labels'] == 1].drop('Labels', axis=1).values
    normal_labels = df[df['Labels'] == 1]['Labels'].values
    anomaly_encodings = df[df['Labels'] == 0].drop(['Labels'], axis=1).values
    anomaly_labels = df[df['Labels'] == 0]['Labels'].values
    X_test, y_test = balance_dataset(normal_encodings, normal_labels, anomaly_encodings, anomaly_labels)
    n_normal = y_test.sum()
    n_anomaly = y_test.shape[0] - n_normal
    sizes = (n_normal, n_anomaly)
    return X_test, y_test, sizes


def create_test_dataset_unbalanced(dataset_id):
    test_f = 'TEST_DATASET_' + str(dataset_id) + '.csv'
    test_path = join(BENCHMARK_DIR, test_f)
    X_test, y_test = get_dataset_from_csv(test_path)
    n_normal = y_test.sum()
    n_anomaly = y_test.shape[0] - n_normal
    sizes = (n_normal, n_anomaly)
    return X_test, y_test, sizes


if __name__ == '__main__':
    #X_train_sup, y_train_sup, sizes_sup = create_train_dataset_supervised(1965)
    #X_test_ub, y_test_ub, sizes_ub = create_test_dataset_unbalanced(1965)
    X_test_b, y_test_b, sizes_b = create_test_dataset_balanced(0)
    print('Finnish')