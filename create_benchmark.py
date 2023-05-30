import os
import pickle
import random
from functools import reduce
from os import listdir
from os.path import join, isfile

import numpy as np
import pandas as pd

WORK_DIR = os.path.dirname(__file__)
FM_DIR = 'feature_maps'
BENCHMARK_DIR = 'benchmark'
DATA_DIR = 'data'
OUTPUTS_DIR = 'outputs'

random.seed(42)


def get_class_names(attributes_file):
    '''
    Make a mapping from class serial number to class names.
    :param attributes_file: The attributes file
    :return: A dictionary with class serial number as key and class name as value
    '''
    attributes_df = pd.read_csv(attributes_file)
    sn_list = attributes_df['sn'].tolist()
    names_list = attributes_df['description'].tolist()
    names_mapping = {}
    for sn, name in zip(sn_list, names_list):
        names_mapping[sn] = name
    return names_mapping


def get_attributes_list(attributes_file, test=0):
    '''
    Specific function for generating attributes for benchmark. First group are single attributes, second group are
    pairs of attributes. For example in a benchmark with 12 attributes, this will generate 12+12*11=144 lists of
    attributes.
    :param attributes_file: a csv file containing the classes and attributes
    :param test: a boolean, if value is 1 generate test data, else generate actual data
    :return: 2 lists, one with attributes indexes (with single attributes and pairs of attributes) and one with
    attributes names.
    '''
    attributes_df = pd.read_csv(attributes_file)
    attributes = attributes_df.columns.tolist()[2:]
    attributes_list = []
    attributes_names_list = []
    att_n = len(attributes)
    if test == 0:
        # only 1 attribute in each group
        for i in range(att_n):
            attributes_list.append([i])
            attributes_names_list.append([attributes[i]])
        # pairs of attributes
        attributes_index = list(range(att_n))
        attributes_index1 = list(range(att_n))
        for i in attributes_index:
            attributes_index1.remove(i)
            for j in attributes_index1:
                attributes_list.append([i, j])
                attributes_names_list.append([attributes[i], attributes[j]])
                attributes_index = list(range(att_n))
            attributes_index1.append(i)
    elif test == 1:
        attributes_list.append([4])
        attributes_list.append([6])
        attributes_list.append([0, 1])
        attributes_list.append([3, 4])
        attributes_names_list.append([attributes[4]])
        attributes_names_list.append([attributes[6]])
        attributes_names_list.append([attributes[0], attributes[1]])
        attributes_names_list.append([attributes[3], attributes[4]])
    return attributes_list, attributes_names_list


def get_classes(df_sub, anomaly_value):
    '''
    This function divides a dataframe with classes and attributes into normal and anomaly data. When several attributes
    are chosen only classes which are found in all attributes are included.
    :param df_sub: A partial dataframe with only relevant attributes.
    :param anomaly_value: the value for anomaly class. If equals 1 than a class with attribute 1 is anomaly, and
    a class with attribute 0 is normal, and vice versa, when anomaly_value=0
    :return: 2 lists of dataframes, one with normal data and one with anomaly data
    '''
    df = df_sub.copy()
    r, c = df.shape
    col_index = list(range(2, c))
    class_values = df.iloc[:, col_index].values
    class_vals_intersection = np.bitwise_or.reduce(class_values, axis=1)
    df['Intersection'] = class_vals_intersection.reshape(r, 1)
    df1 = df[['sn', 'description', 'Intersection']]
    df_normal = df1[df1.loc[:, 'Intersection'] == 1 - anomaly_value]
    df_anomaly = df1[df1.loc[:, 'Intersection'] == anomaly_value]
    return df_normal, df_anomaly


def prepare_df_lists(attributes_file, attributes_index_list):
    '''
    This function prepares lists of normal data and anomaly data.
    The lists contain lists of pkl files.
    The size of the 2 lists are the same. The number of terms in each list is equal to the amount of
    datasets needed.
    attributes_file: a csv file containing the classes and attributes
    attributes_index_list: a list of lists. Each list contains indexes for columns in attributes_file (which are attributes)
    according to this list, the datasets will be constructed. In the first case, in a chosen column the classes with
    value 1 are the normal and all the rest are anomaly, and vice versa in the second case.
    :return:
    normal_df_list: a list containing dataframes with normal data classes (sn, description and intersection or class
    value)
    anomaly_df_list: a list containing dataframes with anomaly data classes (sn, description and intersection or class
    value)
    '''
    normal_df_list = []
    anomaly_df_list = []
    base_attribute_index = 2
    attributes_df = pd.read_csv(attributes_file)
    normal_df = pd.DataFrame()
    anomaly_df = pd.DataFrame()
    for idxs in attributes_index_list:
        base_index = [0, 1]
        col_index = base_index + ((np.array(idxs) + base_attribute_index).tolist())
        # look at sub frame containing: sn, description and relevant attributes
        df_sub = attributes_df.iloc[:, col_index]
        # get the normal and anomaly data (2 tuples, each one with a df of normal classes and list of normal attributes
        # and one with a df of anomaly classes and list of anomaly attributes
        # for first case in which anomaly value is 1
        df_n1, df_a1 = get_classes(df_sub, 1)
        # for second case in which anomaly value is 0
        df_n2, df_a2 = get_classes(df_sub, 0)
        normal_df_list.append(df_n1)
        normal_df_list.append(df_n2)
        anomaly_df_list.append(df_a1)
        anomaly_df_list.append(df_a2)
    return normal_df_list, anomaly_df_list


def get_classes_lists(normal_df_list, anomaly_df_list):
    '''
    This function gets lists of data frames. Each dataframe contains sn (serial number of class) and description
    for classes used in a dataset.
    The outputs are lists of class file names (eg n01440764) and class names (eg 'mud turtle') for normal data
    and anomaly data
    :param normal_df_list: a list of dataframes. Each one with sn, description of several classes. All data is used for
    normal data.
    :param anomaly_df_list: a list of dataframes. Each one with sn, description of several classes. All data is used for
    anomaly data.
    :return:
    n_class_sn: a list containing lists of file names for each dataset, used for normal data.
    n_class_names: a list containing lists of class names for each dataset, used for normal data.
    a_class_sn: a list containing lists of file names for each dataset, used for anomaly data.
    a_class_names: a list containing lists of class names for each dataset, used for anomaly data.
    '''
    n_class_sn = []
    n_class_names = []
    a_class_sn = []
    a_class_names = []
    for df in normal_df_list:
        n_class_sn.append(df.loc[:, 'sn'].tolist())
        n_class_names.append(df.loc[:, 'description'].tolist())
    for df in anomaly_df_list:
        a_class_sn.append(df.loc[:, 'sn'].tolist())
        a_class_names.append(df.loc[:, 'description'].tolist())
    return n_class_sn, n_class_names, a_class_sn, a_class_names


def build_dataset(n_class_sn, a_class_sn, class_names):
    '''
    Go over all pairs of class groups and build dataframes with image file names, encodings, class serial number
    and class names. The outputs are 2 lists containing groups of normal data and anomaly data
    :param n_class_sn: a list of groups of normal classes
    :param a_class_sn: a list of groups of anomaly classes
    :param class_names: a mapping from class serial number to names
    :return:
    n_encoded_df_list: a list of dataframes for normal data
    a_encoded_df_list: a list of dataframes for anomaly data
    '''
    n_encoded_df_list = []
    a_encoded_df_list = []
    i = 0
    for n_file, a_file in zip(n_class_sn, a_class_sn):
        i += 1
        if i % 10 == 0:
            print(f'Get encodings for dataset #{i}')
        n_encoded_df, a_encoded_df = build_encoding_df(n_file, a_file, class_names)
        n_encoded_df_list.append(n_encoded_df)
        a_encoded_df_list.append(a_encoded_df)
    return n_encoded_df_list, a_encoded_df_list


def build_encoding_df(normal_df_list, anomaly_df_list, class_names):
    '''
    Build a dataset from a pair of group classes (normal and anomaly groups).
    :param normal_df_list: a list of classes for normal data
    :param anomaly_df_list: a list of classes for anomaly data
    :param class_names: a mapping from class number to class name
    :return:
    normal_encoding_df: a dataframe with normal data, including: image file name, actual encoding, class serial number,
    class name. Names of columns are: FileName, Encoding, ClassNumber, ClassName
    anomaly_encoding_df: a dataframe with anomaly data, including: image file name, actual encoding, class serial number,
    class name. Names of columns are: FileName, Encoding, ClassNumber, ClassName
    '''
    normal_encoding_df = pd.DataFrame()
    anomaly_encoding_df = pd.DataFrame()
    dir = join(WORK_DIR, FM_DIR)
    files = [f for f in listdir(dir) if isfile(join(dir, f))]
    for file in files:
        for enc_file in normal_df_list:
            if file == enc_file + '.pkl':
                path = join(dir, file)
                fname = open(path, 'rb')
                df_normal1 = pickle.load(fname)
                fname.close()
                df_normal1 = df_normal1.drop(['FeatureMap'], axis=1)
                df_normal1['FeatureMapFile'] = file
                df_normal1['FeatureMapIndex'] = df_normal1.index
                df_normal1['ClassNumber'] = enc_file
                df_normal1['ClassName'] = class_names[enc_file]
                normal_encoding_df = pd.concat([normal_encoding_df, df_normal1], axis=0)
        for enc_file in anomaly_df_list:
            if file == enc_file + '.pkl':
                path = join(dir, file)
                fname = open(path, 'rb')
                df_anomaly1 = pickle.load(fname)
                fname.close()
                df_anomaly1 = df_anomaly1.drop(['FeatureMap'], axis=1)
                df_anomaly1['FeatureMapFile'] = file
                df_anomaly1['FeatureMapIndex'] = df_anomaly1.index
                df_anomaly1['ClassNumber'] = enc_file
                df_anomaly1['ClassName'] = class_names[enc_file]
                anomaly_encoding_df = pd.concat([anomaly_encoding_df, df_anomaly1], axis=0)
    normal_encoding_df.reset_index(drop=True, inplace=True)
    anomaly_encoding_df.reset_index(drop=True, inplace=True)
    return normal_encoding_df, anomaly_encoding_df


def dataset_split(df_normal, df_anomaly, n_train_percent=0.7, a_test_percent=0.3):
    min_anomaly_sup_train_size = 500
    # Need to check if df_normal or df_anomly are empty!!!!
    if (df_normal.shape[0] == 0) | (df_anomaly.shape[0] == 0):
        dataset_ok = 0
        df_normal_train = np.nan
        df_test = np.nan
        df_anomaly_sup_train = np.nan
    else:
        dataset_ok = 1
        n_samples_normal = df_normal.shape[0]
        n_samples_anomaly = df_anomaly.shape[0]
        normal_train_size = int(np.round(n_samples_normal * n_train_percent))
        anomaly_test_size = int(np.round(n_samples_anomaly * a_test_percent))
        anomaly_sup_train_size = n_samples_anomaly - anomaly_test_size
        if anomaly_sup_train_size < min_anomaly_sup_train_size:  # size of sup train dataset is smaller than min
            dataset_ok = 0
            df_normal_train = np.nan
            df_test = np.nan
            df_anomaly_sup_train = np.nan
        else:
            # get train dataset
            normal_idx = random.sample(range(n_samples_normal), normal_train_size)
            df_normal_train = df_normal.iloc[normal_idx, :]
            df_normal_train = df_normal_train.sample(frac=1).reset_index(drop=True)
            df_normal_train['IsNormal'] = 1
            # get test dataset
            # test from normal data
            df_normal_test = df_normal.drop(index=normal_idx).reset_index(drop=True)
            df_normal_test['IsNormal'] = 1
            # test from anomaly data
            anomaly_test_idx = random.sample(range(n_samples_anomaly), anomaly_test_size)
            df_anomaly_test = df_anomaly.iloc[anomaly_test_idx, :]
            df_anomaly_test.sample(frac=1).reset_index(drop=True, inplace=True)
            df_anomaly_test['IsNormal'] = 0
            df_test = pd.concat([df_normal_test, df_anomaly_test], axis=0).reset_index(drop=True)
            df_test = df_test.sample(frac=1).reset_index(drop=True)
            # get anomaly data for supervised train
            df_anomaly_sup_train = df_anomaly.drop(index=anomaly_test_idx)
            df_anomaly_sup_train = df_anomaly_sup_train.sample(frac=1).reset_index(drop=True)
            df_anomaly_sup_train['IsNormal'] = 0
    return dataset_ok, df_normal_train, df_test, df_anomaly_sup_train


def create_benchmark_table(ne_df_list, ae_df_list, class_names):
    col_names = ['DatasetName', 'DatasetSource',
                 'DataType', 'PartitonMethod', 'AnomalyClassNames', 'TrainSize', 'TestSize',
                 'AnomalyTestPercent', 'BalancedTestSize', 'TrainLink', 'TestLink', 'SupTrainLink',
                 'UpperBoundMean', 'UpperBoundSTD', 'DatasetOk']
    benchmark_df = pd.DataFrame(columns=col_names)
    train_df_list = []
    test_df_list = []
    supervised_train_df_list = []
    ds_size = len(ne_df_list)
    anomaly_fraction = 0.3
    for i in range(ds_size):
        benchmark_df.loc[i, 'DatasetSource'] = 'ImageNet.ILSVRC'
        benchmark_df.loc[i, 'DataType'] = 'Images'
        benchmark_df.loc[i, 'AnomalyTestPercent'] = anomaly_fraction
        benchmark_df.loc[i, 'DatasetName'] = 'ILSVRC.' + str(i)
        benchmark_df.loc[i, 'AnomalyClassNames'] = class_names[i]
        benchmark_df.loc[i, 'PartitonMethod'] = reduce(lambda x, y: x + ', ' + y, attribute_df.loc[i, 'AttributeNames'])
        dataset_ok, train_df, test_df, supervised_train_df = dataset_split(ne_df_list[i],
                                                                           ae_df_list[i],
                                                                           n_train_percent=0.7,
                                                                           a_test_percent=anomaly_fraction)
        train_df_list.append(train_df)
        test_df_list.append(test_df)
        supervised_train_df_list.append(supervised_train_df)
        if dataset_ok:
            benchmark_df.loc[i, 'DatasetOk'] = 1
            # Save unsupervised train data to file
            train_file_name = 'TRAIN_DATASET_' + str(i) + '.csv'
            path = join(WORK_DIR, BENCHMARK_DIR, train_file_name)
            # df = train_df.copy()
            # df[['FileName', 'ClassNumber', 'ClassName']].drop
            train_df.to_csv(path)
            benchmark_df.loc[i, 'TrainLink'] = train_file_name
            # Save unsupervised test data to file
            test_file_name = 'TEST_DATASET_' + str(i) + '.csv'
            path = join(WORK_DIR, BENCHMARK_DIR, test_file_name)
            test_df.to_csv(path)
            benchmark_df.loc[i, 'TestLink'] = test_file_name
            # Save supervised train data to file
            sup_train_file_name = 'SUP_TRAIN_DATASET_' + str(i) + '.csv'
            path = join(WORK_DIR, BENCHMARK_DIR, sup_train_file_name)
            supervised_train_df.to_csv(path)
            benchmark_df.loc[i, 'SupTrainLink'] = sup_train_file_name
            # See balanced dataset size
            benchmark_df.loc[i, 'TrainSize'] = train_df.shape[0]
            benchmark_df.loc[i, 'TestSize'] = test_df.shape[0]
            benchmark_df.loc[i, 'AnomalyTestPercent'] = anomaly_fraction
            normal_test_size = np.round(ne_df_list[i].shape[0] * 0.3)
            anomaly_test_size = np.round(ae_df_list[i].shape[0] * anomaly_fraction)
            if normal_test_size > anomaly_test_size:
                balanced_size = str(anomaly_test_size) + ' A'
            else:
                balanced_size = str(normal_test_size) + ' N'
            benchmark_df.loc[i, 'BalancedTestSize'] = balanced_size
        else:
            benchmark_df.loc[i, 'DatasetOk'] = 0
    return benchmark_df, train_df_list, test_df_list, supervised_train_df_list


if __name__ == '__main__':
    if not os.path.isdir(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.isdir(OUTPUTS_DIR):
        os.mkdir(OUTPUTS_DIR)
    if not os.path.isdir(BENCHMARK_DIR):
        os.mkdir(BENCHMARK_DIR)

    input_file = join(WORK_DIR, DATA_DIR, 'objects.csv')
    # create a mapping from class serial number to class names
    class_names = get_class_names(input_file)
    # create list of attributes to choose datasets from
    attrb_index1, attrb_name1 = get_attributes_list(input_file, test=0)
    # Multiply attributes to add conjugate classes
    attrb_name = []
    attrb_index = []
    for att_i, att_n in zip(attrb_index1, attrb_name1):
        attrb_name.append(att_n)
        attrb_name.append(att_n + ['Inverted'])
        attrb_index.append(att_i)
        attrb_index.append(att_i)
    # save attributes
    attribute_df = pd.DataFrame()
    attribute_df['AttributeIndex'] = attrb_index
    attribute_df['AttributeNames'] = attrb_name
    att_file = join(WORK_DIR, OUTPUTS_DIR, 'attributes.pkl')
    attribute_file = open(att_file, 'wb')
    pickle.dump(attribute_df, attribute_file)
    attribute_file.close()
    # create list of dataframes with class numbers, class names and a attribute value or an
    # intersection (for multi attributes sets) value.
    dfn_list, dfa_list = prepare_df_lists(input_file, attrb_index1)
    # Get class groups for each dataset, the length of lists is the amount of datasets.
    # Output are 4 lists:
    # 1) normal class number list
    # 2) normal class names list
    # 3) anomaly class number list
    # 4) anomaly class names list
    n_class_sn_list, n_class_names_list, a_class_sn_list, a_class_names_list = get_classes_lists(dfn_list, dfa_list)
    # Get actual encoding for each dataset. Each item in list is a dataframe with image file name,
    # actual encoding, class serial number, class name.
    n_enc_df_list, a_enc_df_list = build_dataset(n_class_sn_list, a_class_sn_list, class_names)
    # dataset_file = open('dataset.pkl', 'wb')
    # pickle.dump((n_enc_df_list, a_enc_df_list), dataset_file)
    # dataset_file.close()
    # Build benchmark df
    benchmark_df, train_u_df, test_u_df, train_s_df = create_benchmark_table(n_enc_df_list, a_enc_df_list,
                                                                             a_class_names_list)
    image_benchmark_file = join(WORK_DIR, OUTPUTS_DIR, 'Image_Benchmark.csv')
    benchmark_df.to_csv('image_benchmark_file')
    image_benchmark_data_file = join(WORK_DIR, OUTPUTS_DIR, 'Image_Benchmark.pkl')
    fname = open(image_benchmark_data_file, 'wb')
    pickle.dump((benchmark_df, train_u_df, test_u_df, train_s_df), fname)
    fname.close()
    print('Finnish')
