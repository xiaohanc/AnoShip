import os
import pickle as pk

import numpy as np
import pandas as pd
import time

pth = './'

init_data = pd.read_csv('train.csv').to_numpy()
test_data = pd.read_csv('test.csv').to_numpy()
test_label = pd.read_csv('test_label.csv').to_numpy()

init_time = pd.date_range(start='2024-01-01 00:00:00', periods=init_data.shape[0], freq='min')
test_time = pd.date_range(start=init_time[-1], periods=test_data.shape[0], freq='min')

print('  Dumping pickle files...')
with open(pth + 'PSM.pk', 'wb') as file:
    pk.dump({'init_data': [init_data[:, 1:]], 'test_data': [test_data[:, 1:]],
             'test_label': [test_label[:, 1:]], 'init_time': [init_time], 'test_time': [test_time]}, file)

print('Done')