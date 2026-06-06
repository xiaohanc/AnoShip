import numpy as np
import pandas as pd
import pickle as pk
from mstdf_ad.utils.dwt import dwt
from sklearn.preprocessing import StandardScaler


def getTimeEmbedding(time):
    df = pd.DataFrame(time, columns=['time'])
    df['time'] = pd.to_datetime(df['time'])

    df['second'] = df['time'].apply(lambda row: row.second / 59 - 0.5)
    df['minute'] = df['time'].apply(lambda row: row.minute / 59 - 0.5)
    df['hour'] = df['time'].apply(lambda row: row.hour / 23 - 0.5)
    df['weekday'] = df['time'].apply(lambda row: row.weekday() / 6 - 0.5)
    df['day'] = df['time'].apply(lambda row: row.day / 30 - 0.5)
    df['month'] = df['time'].apply(lambda row: row.month / 365 - 0.5)

    return df[['second', 'minute', 'hour', 'weekday', 'day', 'month']].values


def getData(path='./dataset/', dataset='PSM', train_rate=0.8, entities='all', wavelet='db4', level=5, window_length=51,
            polyorder=3, p=3):
    with open(path + dataset + '/' + dataset + '.pk', 'rb') as file:
        data = pk.load(file)

    entity_num = len(data['init_data'])
    if entities != 'all':
        int_list = [int(x) for x in entities.split(',')]
        entity_num = len(int_list)
        for key in data:
            data[key] = [data[key][entity] for entity in int_list]

    init_datas = data['init_data']
    init_times = data['init_time']

    test_datas = data['test_data']
    test_times = data['test_time']
    test_label = data['test_label']

    init_p = 0
    scaler = StandardScaler()
    all_init_data = np.concatenate(init_datas, axis=0)

    if dataset == 'MSL':
        all_test_data = np.concatenate(test_datas, axis=0)
        zero_columns = np.all(all_init_data == 0, axis=0) & np.all(all_test_data == 0, axis=0)
        all_init_data = all_init_data[:, ~zero_columns]

    if dataset == 'MSL' or dataset == 'SMAP':
        scaled_value = all_init_data[:, 0].reshape(-1, 1)
        scaler.fit(scaled_value)
        scaled_value = pd.DataFrame(scaler.transform(scaled_value)).fillna(0).values
        variance_value = np.var(scaled_value, axis=0)
        init_p = p * variance_value
        zeros_array = np.zeros(all_init_data.shape[1] - 1)
        init_p = np.concatenate((init_p, zeros_array))
    else:
        scaler.fit(all_init_data)
        scaled_init_data = pd.DataFrame(scaler.transform(all_init_data)).fillna(0).values
        variance_value = np.var(scaled_init_data, axis=0)
        init_p = np.maximum(p * variance_value, 1)

    init_data, init_time, init_label = [], [], []
    test_data, test_time = [], []
    train_data, train_time, train_label = [], [], []
    valid_data, valid_time, valid_label = [], [], []
    init_S, test_S, train_S, valid_S = [], [], [], []
    init_T, test_T, train_T, valid_T = [], [], [], []

    for entity, entity_time in zip(init_datas, init_times):

        if dataset == 'MSL':
            entity = entity[:, ~zero_columns]

        if dataset == 'SWaT':
            entity = entity[100000:, :]
            entity_time = entity_time[100000:]

        if dataset == 'MSL' or dataset == 'SMAP':
            scaled_entity = entity
            scaled_value = entity[:, 0].reshape(-1, 1)
            scaled_value = pd.DataFrame(scaler.transform(scaled_value)).fillna(0).values
            scaled_entity[:, 0] = scaled_value.reshape(-1)
            T = np.zeros_like(entity)
            T_value, _ = dwt(scaled_value, wavelet, level)
            T[:, 0] = T_value.reshape(-1)
            S = entity - T
        else:
            scaled_entity = pd.DataFrame(scaler.transform(entity)).fillna(0).values
            T, S = dwt(scaled_entity, wavelet, level, True, window_length, polyorder)

        entity_time = getTimeEmbedding(entity_time)
        entity_label = np.zeros((len(entity), 1))

        init_data.append(scaled_entity)
        init_S.append(S)
        init_T.append(T)
        init_time.append(entity_time)
        init_label.append(entity_label)

        if len(scaled_entity) <= 100000:
            train_data.append(scaled_entity[:int(train_rate * len(scaled_entity)), :])
            train_time.append(entity_time[:int(train_rate * len(scaled_entity)), :])
            train_label.append(entity_label[:int(train_rate * len(scaled_entity)), :])
            train_S.append(S[:int(train_rate * len(scaled_entity)), :])
            train_T.append(T[:int(train_rate * len(scaled_entity)), :])

            valid_data.append(scaled_entity[int(train_rate * len(scaled_entity)):, :])
            valid_time.append(entity_time[int(train_rate * len(scaled_entity)):, :])
            valid_label.append(entity_label[int(train_rate * len(scaled_entity)):, :])
            valid_S.append(S[int(train_rate * len(scaled_entity)):, :])
            valid_T.append(T[int(train_rate * len(scaled_entity)):, :])

        else:
            train_data.append(scaled_entity[:int(train_rate * len(scaled_entity) // 2), :])
            train_time.append(entity_time[:int(train_rate * len(scaled_entity) // 2), :])
            train_label.append(entity_label[:int(train_rate * len(scaled_entity) // 2), :])
            train_S.append(S[:int(train_rate * len(scaled_entity) // 2), :])
            train_T.append(T[:int(train_rate * len(scaled_entity) // 2), :])

            valid_data.append(scaled_entity[int(train_rate * len(scaled_entity) // 2):int(len(scaled_entity) // 2), :])
            valid_time.append(entity_time[int(train_rate * len(scaled_entity) // 2):int(len(scaled_entity) // 2), :])
            valid_label.append(entity_label[int(train_rate * len(scaled_entity) // 2):int(len(scaled_entity) // 2), :])
            valid_S.append(S[int(train_rate * len(scaled_entity) // 2):int(len(scaled_entity) // 2), :])
            valid_T.append(T[int(train_rate * len(scaled_entity) // 2):int(len(scaled_entity) // 2), :])

            train_data.append(
                scaled_entity[int(len(scaled_entity) // 2):int((train_rate + 1) * len(scaled_entity) // 2), :])
            train_time.append(
                entity_time[int(len(scaled_entity) // 2):int((train_rate + 1) * len(scaled_entity) // 2), :])
            train_label.append(
                entity_label[int(len(scaled_entity) // 2):int((train_rate + 1) * len(scaled_entity) // 2), :])
            train_S.append(S[int(len(scaled_entity) // 2):int((train_rate + 1) * len(scaled_entity) // 2), :])
            train_T.append(T[int(len(scaled_entity) // 2):int((train_rate + 1) * len(scaled_entity) // 2), :])

            valid_data.append(scaled_entity[int((train_rate + 1) * len(scaled_entity) // 2):, :])
            valid_time.append(entity_time[int((train_rate + 1) * len(scaled_entity) // 2):, :])
            valid_label.append(entity_label[int((train_rate + 1) * len(scaled_entity) // 2):, :])
            valid_S.append(S[int((train_rate + 1) * len(scaled_entity) // 2):, :])
            valid_T.append(T[int((train_rate + 1) * len(scaled_entity) // 2):, :])

    for entity, entity_time in zip(test_datas, test_times):

        if dataset == 'MSL':
            entity = entity[:, ~zero_columns]

        if dataset == 'MSL' or dataset == 'SMAP':
            scaled_entity = entity
            scaled_value = entity[:, 0].reshape(-1, 1)
            scaled_value = pd.DataFrame(scaler.transform(scaled_value)).fillna(0).values
            scaled_entity[:, 0] = scaled_value.reshape(-1)
            T = np.zeros_like(entity)
            T_value, _ = dwt(scaled_value, wavelet, level)
            T[:, 0] = T_value.reshape(-1)
            S = entity - T
        else:
            scaled_entity = pd.DataFrame(scaler.transform(entity)).fillna(0).values
            T, S = dwt(scaled_entity, wavelet, level, True, window_length, polyorder)

        entity_time = getTimeEmbedding(entity_time)

        test_data.append(scaled_entity)
        test_S.append(S)
        test_T.append(T)
        test_time.append(entity_time)

    data = {
        'init_p': init_p,
        'feature_num': init_data[0].shape[1],
        'time_num': init_time[0].shape[1],
        'entity_num': entity_num,
        'train_num': np.concatenate(train_data, axis=0).shape[0],

        'train': {
            'data': train_data, 'time': train_time, 'label': train_label,
            'S': train_S, 'T': train_T
        },

        'valid': {
            'data': valid_data, 'time': valid_time, 'label': valid_label,
            'S': valid_S, 'T': valid_T
        },

        'init': {
            'data': init_data, 'time': init_time, 'label': init_label,
            'S': init_S, 'T': init_T
        },

        'test': {
            'data': test_data, 'time': test_time, 'label': test_label,
            'S': test_S, 'T': test_T
        },
    }

    return data
