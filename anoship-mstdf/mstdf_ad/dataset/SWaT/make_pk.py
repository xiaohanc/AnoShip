import numpy as np
import pandas as pd
import pickle as pk

pth = './'

pd_init = pd.read_csv('SWaT_Dataset_Normal_v1.csv', skiprows=1)
pd_test = pd.read_csv('SWaT_Dataset_Attack_v0.csv', skiprows=1)

channels = pd_init.columns[1:-1]
init_data = pd_init[pd_init.columns[1:-1]].to_numpy()
test_data = pd_test[pd_test.columns[1:-1]].to_numpy()
test_label = np.array([ float(label!= 'Normal' ) for label in pd_test["Normal/Attack"].values]).reshape(-1, 1)

init_time = np.array(pd.to_datetime(pd_init[' Timestamp'].str.strip(), format='%d/%m/%Y %I:%M:%S %p'))
test_time = np.array(pd.to_datetime(pd_test[' Timestamp'].str.strip(), format='%d/%m/%Y %I:%M:%S %p'))

print('  Dumping pickle files...')
with open('SWaT.pk', 'wb') as file:
    pk.dump({'init_data': [init_data], 'init_time': [init_time],
             'test_data': [test_data], 'test_time': [test_time], 'test_label': [test_label]}, file)

print('Done')
