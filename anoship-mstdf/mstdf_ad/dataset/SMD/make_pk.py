import os
import pickle as pk

import numpy as np
import pandas as pd

pth = './'

# 定义自定义的排序键函数
def natural_sort_key(s):
    import re
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

# 对文件名列表进行自然排序
ent_names = sorted(os.listdir(pth + 'train'), key=natural_sort_key)

init_data, test_data, test_label = [], [], []
init_time, test_time = [], []

for ent_name in ent_names:
    init_data.append(pd.read_csv(pth + 'train/' + ent_name, header=None).to_numpy())
    test_data.append(pd.read_csv(pth + 'test/' + ent_name, header=None).to_numpy())
    test_label.append(np.squeeze(pd.read_csv(pth + 'test_label/' + ent_name, header=None).to_numpy()).reshape(-1, 1))

start_time = '2024-01-01 00:00:00'

for init_entity, test_entity in zip(init_data, test_data):
    init_time.append(pd.date_range(start=start_time, periods=init_entity.shape[0], freq='min').to_numpy().reshape(-1, 1))
    start_time = init_time[-1][-1][-1]
    test_time.append(pd.date_range(start=start_time, periods=test_entity.shape[0], freq='min').to_numpy().reshape(-1, 1))
    start_time = init_time[-1][-1][-1]

print('  Dumping pickle files...')
with open(pth + 'SMD.pk', 'wb') as file:
    pk.dump({'init_data': init_data, 'test_data': test_data, 'test_label': test_label, 'init_time': init_time, 'test_time': test_time}, file)

print('Done')