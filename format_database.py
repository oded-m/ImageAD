import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('Image_Benchmark.csv')
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
    df_new.to_csv('Image_Benchmark_Table.csv')
    print('Finnish')
