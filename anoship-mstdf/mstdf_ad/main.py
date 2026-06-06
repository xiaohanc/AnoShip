import argparse
import torch
import numpy as np
from mstdf_ad.exp.exp import Exp
# from exp_NSMCEV.exp import exp_NSMCEV  # (module not included in this repo)
from mstdf_ad.utils.seed import setSeed
from mstdf_ad.utils.getentitylist import getEntityList

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='PSM', help='dataset')
    parser.add_argument('--model_name', type=str, default='MSTDF', help='model')
    parser.add_argument('--data_dir', type=str, default='./dataset/', help='path of the data')

    parser.add_argument('--itr', type=int, default=3, help='num of evaluation')
    parser.add_argument('--epochs', type=int, default=64, help='epoch of train')
    parser.add_argument('--patience', type=int, default=3, help='patience of early stopping')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of data')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate of optimizer')

    parser.add_argument('--train_rate', type=float, default=0.8, help='rate of train set')
    parser.add_argument('--window_size', type=int, default=64, help='size of sliding window')
    parser.add_argument('--step_size', type=int, default=16, help='move step size of sliding window')
    parser.add_argument('--entities', type=str, default='all', help='index of entities such as \'1,2,3,4,5\'')
    parser.add_argument('--each_entity', type=bool, default=False, help='whether to train every entity')

    parser.add_argument('--model_dim', type=int, default=512, help='dimension of hidden layer')
    parser.add_argument('--ff_dim', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--atten_dim', type=int, default=64, help='dimension of various attention')
    parser.add_argument('--hidden_dim', type=int, default=5, help='dimension of potential spatial')
    parser.add_argument('--kernel_size', type=int, default=16, help='size of convolution kernel')
    parser.add_argument('--stride', type=int, default=8, help='size of convolution step')

    parser.add_argument('--block_num', type=int, default=2, help='num of various block')
    parser.add_argument('--head_num', type=int, default=8, help='num of attention head')
    parser.add_argument('--dropout', type=float, default=0.4, help='dropout')

    parser.add_argument('--wavelet', type=str, default='db4', help='wavelet')
    parser.add_argument('--level', type=int, default=5, help='level')
    parser.add_argument('--window_length', type=int, default=51, help='sliding window length of savgol_filter')
    parser.add_argument('--polyorder', type=int, default=3, help='polyorder order')

    parser.add_argument('--p', type=float, default=1, help='peak value of trend disturbance')
    parser.add_argument('--epsilon', type=float, default=0.2, help='noise rate')
    parser.add_argument('--anomaly_ratio', type=float, default=1, help='anomalyratio')

    parser.add_argument('--random_seed', type=int, default=1234, help='random seed')
    parser.add_argument('--gpu_id', type=int, default=0, help='device ids of gpus')

    config = vars(parser.parse_args())
    setSeed(config['random_seed'])
    torch.cuda.set_device(config['gpu_id'])

    F1_itr = []
    precision_itr = []
    recall_itr = []
    entities = config['entities']
    for ii in range(config['itr']):
        config['entities'] = entities
        if config['each_entity'] == True:
            F1_entity = []
            precision_entity = []
            recall_entity = []
            entity_list = getEntityList(config['data_dir'], config['dataset'], config['entities'])
            for i in entity_list:
                print('entity number:', i)
                config['entities'] = str(i)
                exp = Exp(config)
                exp.train()
                F1_score, precision, recall = exp.test()
                F1_entity.append(F1_score)
                precision_entity.append(precision)
                recall_entity.append(recall)
            F1_itr.append(np.mean(F1_entity))
            precision_itr.append(np.mean(precision_entity))
            recall_itr.append(np.mean(recall_entity))
            print("\n=============== " + config['dataset'] + " ===============")
            print(f"P: {precision_itr[-1]:.4f} || R: {recall_itr[-1]:.4f} || F1: {F1_itr[-1]:.4f}")
            print("=============== " + config['dataset'] + " ===============\n")
        else:
            exp = Exp(config)
            exp.train()
            F1_score, precision, recall = exp.test()
            F1_itr.append(F1_score)
            precision_itr.append(precision)
            recall_itr.append(recall)

    avg_F1_itr = np.mean(F1_itr)
    avg_precision_itr = np.mean(precision_itr)
    avg_recall_itr = np.mean(recall_itr)

    print("\n=============== " + config['dataset'] + " ===============")
    print(f"P: {avg_precision_itr:.4f} || R: {avg_recall_itr:.4f} || F1: {avg_F1_itr:.4f}")
    print("=============== " + config['dataset'] + " ===============\n")