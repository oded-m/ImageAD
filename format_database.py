import pandas as pd
import os

WORK_DIR = os.path.dirname(__file__)
FM_DIR = 'feature_maps'
BENCHMARK_DIR = os.path.join(WORK_DIR, 'benchmark')
DATA_DIR = os.path.join(WORK_DIR, 'data')
OUTPUTS_DIR = os.path.join(WORK_DIR, 'outputs')

if __name__ == '__main__':
    path = os.path.join(OUTPUTS_DIR, 'Image_Benchmark.csv')
    df = pd.read_csv(path)
    df = df[df['DatasetOk'] == 1]
    col_names = ['DatasetName', 'DatasetSource', 'DataType', 'PartitonMethod',
                 'AnomalyClassNames', 'TrainSize', 'TestSize',
                 'AnomalyTestPercent', 'BalancedTestSize']
    df_new = df.loc[:, col_names]
    r = df_new.shape[0]
    names = []
    for i in range(r):
        names.append('ILSVRC.' + str(i))
    df_new['DatasetName'] = names
    df_new.reset_index(drop=True, inplace=True)
    path = os.path.join(OUTPUTS_DIR, 'Image_Benchmark_Table.csv')
    df_new.to_csv(path)
    print('Finnish')
