import numpy as np
import pandas as pd
import pickle as pk

pth = './'

def get_neighbor_idx(total_len, target_idx, H=720):
    return [np.max([0, target_idx - H]), np.min([total_len, target_idx + H + 1])]

def remove_outliers(data, threshold=3, w=1441):
    def remove_value(idx):
        start_idx, end_idx = get_neighbor_idx(len(data), idx, w//2)
        data_mean = np.mean(data[start_idx:end_idx, :], axis=0)
        data_std = np.std(data[start_idx:end_idx, :], axis=0)
        data[idx][(data[idx] - data_mean) > threshold * data_std] = data_mean[(data[idx] - data_mean) > threshold * data_std]
        return data[idx]

    data=data.reshape(-1, 1)
    idx_list = np.arange(len(data))
    data = np.array(list(map(remove_value, idx_list)))
    return data.reshape(-1)

labeled_anomalies = pd.read_csv(pth + 'labeled_anomalies.csv')

data_dims = {'SMAP': 25, 'MSL': 55}

for smap_or_msl in ['SMAP']:
    print(f'Creating dataset for {smap_or_msl}')
    init_data = []
    init_time = []
    test_data = []
    test_time = []
    test_label = []
    start_time = '2024-01-01 00:00:00'
    total_anomaly_points = 0
    for i in range(len(labeled_anomalies)):
        print(f'  -> {labeled_anomalies["chan_id"][i]} ({i + 1} / {len(labeled_anomalies)})')
        if labeled_anomalies['spacecraft'][i] == smap_or_msl:
            # load corresponding .npy file in test and train
            np_trn = np.load(pth + 'train/' + labeled_anomalies['chan_id'][i] + '.npy')
            assert np_trn.shape[-1] == data_dims[smap_or_msl]
            # np_trn[:,0] = remove_outliers(np_trn[:,0])
            init_data.append(np_trn)
            init_time.append(
                pd.date_range(start=start_time, periods=np_trn.shape[0], freq='min').to_numpy().reshape(-1, 1))
            start_time = init_time[-1][-1][-1]

            np_tst = np.load(pth + 'test/' + labeled_anomalies['chan_id'][i] + '.npy')
            assert np_tst.shape[-1] == data_dims[smap_or_msl]
            test_data.append(np_tst)
            test_time.append(
                pd.date_range(start=start_time, periods=np_tst.shape[0], freq='min').to_numpy().reshape(-1, 1))
            start_time = init_time[-1][-1][-1]

            labs = labeled_anomalies['anomaly_sequences'][i]
            labs_s = labs.replace('[', '').replace(']', '').replace(' ', '').split(',')
            labs_i = [[int(labs_s[i]), int(labs_s[i + 1])] for i in range(0, len(labs_s), 2)]

            assert labeled_anomalies['num_values'][i] == len(np_tst)
            y_lab = np.zeros(len(np_tst))
            for sec in labs_i:
                y_lab[sec[0]:sec[1]] = 1
                total_anomaly_points += sec[1] - sec[0]
            test_label.append(y_lab.reshape(-1, 1))

    print('  Dumping pickle files...')
    with open(pth + smap_or_msl + '.pk', 'wb') as file:
        pk.dump({'init_data': init_data, 'test_data': test_data, 'test_label': test_label, 'init_time': init_time, 'test_time': test_time}, file)

print('Done')

