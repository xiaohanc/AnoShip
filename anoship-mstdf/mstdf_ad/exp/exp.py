import os
import random
from math import exp
from time import time

import datetime as dt
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from mstdf_ad.data.dataset import Dataset
from mstdf_ad.data.preprocess import getData
from mstdf_ad.model.MSTDF import MSTDF
from mstdf_ad.utils.earlystop import EarlyStop
from mstdf_ad.utils.evaluate import evaluate
from mstdf_ad.utils.getKLContributions import kl_divergence_contributions


class Exp:
    def __init__(self, config):
        self.__dict__.update(config)

        self.all_res = f'{config}\n'
        self.model_dir = f'./results/{self.dataset}/{self.model_name}/'

        self._get_data()
        self._get_model()

        self.free_bits = self.hidden_dim / 10
        self.beta = [0.]
        self.last_KL_divergence = 0
        self.T = 64

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def _get_data(self):
        data = getData(
            path=self.data_dir,
            dataset=self.dataset,
            train_rate=self.train_rate,
            entities=self.entities,
            wavelet=self.wavelet,
            level=self.level,
            window_length=self.window_length,
            polyorder=self.polyorder,
            p=self.p
        )

        self.feature_num = data['feature_num']
        self.time_num = data['time_num']
        self.entity_num = data['entity_num']
        self.C = np.ceil(np.log2(self.hidden_dim)) * np.ceil(np.log2(data['train_num']))
        self.init_p = data['init_p']

        if self.dataset == 'MSL' or self.dataset == 'SMAP':
            zeros_array = np.zeros(self.feature_num - 1)
            self.epsilon = np.concatenate((np.array(self.epsilon).reshape(-1), zeros_array))
            self.epsilon = torch.tensor(self.epsilon)

        print('\ndata shape: ')
        for k, v in data.items():
            if isinstance(v, dict):
                print('------', k, '------')
                for kk, vv in v.items():
                    vv_all = np.concatenate(vv, axis=0)
                    print(kk, ': ', vv_all.shape)

            elif not isinstance(v, np.ndarray):
                print(k, ':', v)

        self.train_set = Dataset(
            data=data['train'],
            window_size=self.window_size,
            step_size=self.step_size
        )
        self.valid_set = Dataset(
            data=data['valid'],
            window_size=self.window_size,
            step_size=self.step_size
        )
        self.init_set = Dataset(
            data=data['init'],
            window_size=self.window_size,
            step_size=self.step_size
        )
        self.test_set = Dataset(
            data=data['test'],
            window_size=self.window_size,
            step_size=self.step_size
        )

        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, drop_last=False)
        self.valid_loader = DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=False, drop_last=False)
        self.init_loader = DataLoader(self.init_set, batch_size=self.batch_size, shuffle=False, drop_last=False)
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, drop_last=False)

    def _get_model(self):
        print('\ncuda is available', torch.cuda.is_available())
        print('\ndevice name', torch.cuda.get_device_name())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('\ndevice:', self.device)

        if self.model_name == 'MSTDF':
            self.model = MSTDF(
                dataset=self.dataset,
                window_size=self.window_size,
                model_dim=self.model_dim,
                ff_dim=self.ff_dim,
                atten_dim=self.atten_dim,
                feature_num=self.feature_num,
                time_num=self.time_num,
                block_num=self.block_num,
                head_num=self.head_num,
                dropout=self.dropout,
                device=self.device,
                hidden_dim=self.hidden_dim,
                kernel_size=self.kernel_size,
                stride=self.stride,
                epsilon=self.epsilon,
            ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        self.early_stopping = EarlyStop(patience=self.patience, path=self.model_dir + self.dataset + '_model.pkl')
        self.criterion = nn.MSELoss(reduction='mean')

    def calculateBeta(self, KL_divergence, last_KL_divergence, beta):
        dis_last = abs(last_KL_divergence - self.C)
        dis_now = abs(KL_divergence - self.C)
        rate = 0
        if KL_divergence < self.C:
            if KL_divergence < last_KL_divergence and last_KL_divergence < self.C:
                d_dis = dis_now - dis_last
                d_dis *= 10
                rate = exp(-(d_dis * self.T))
                if random.random() >= rate:
                    if len(beta) > 1:
                        beta.pop()
            elif KL_divergence > last_KL_divergence:
                d_dis = dis_last - dis_now
                d_dis *= 10
                rate = exp(-(d_dis * self.T))
                if random.random() >= rate:
                    if beta[-1] != 1:
                        beta.append(min(1, beta[-1] + self.min_step))
        else:
            if beta[-1] != 1:
                beta.append(min(1, beta[-1] + dis_now * self.min_step))
        return beta

    def _process_one_batch(self, batch, train, valid):
        batch_data = batch['data'].float().to(self.device)
        batch_time = batch['time'].float().to(self.device)
        batch_S = batch['S'].float().to(self.device)
        batch_T = batch['T'].float().to(self.device)

        if train:
            mu_z_x, mu_z_y, logvar_z_x, logvar_z_y, z_x, stable, trend, recon = self.model(batch_data, batch_time,
                                                                                           self.init_p)
            mu_z = mu_z_x.reshape(-1, self.hidden_dim)
            logvar_z = logvar_z_x.reshape(-1, self.hidden_dim)
            KL_divergence = torch.mean(torch.sum(-0.5 * (1 + logvar_z - torch.exp(logvar_z) - mu_z ** 2), 1))

            recon_freq = torch.fft.fftn(recon, dim=(1, 2))
            batch_data_freq = torch.fft.fftn(batch_data, dim=(1, 2))

            mse_real = torch.mean((recon_freq.real - batch_data_freq.real) ** 2)
            mse_imag = torch.mean((recon_freq.imag - batch_data_freq.imag) ** 2)

            recon_loss = self.criterion(recon, batch_data) + self.criterion(stable, batch_S) + self.criterion(trend,
                                                                                                              batch_T) + self.criterion(
                mu_z_x, mu_z_y) + self.criterion(logvar_z_x, logvar_z_y) + mse_real + mse_imag

            if not valid:
                self.beta = self.calculateBeta(KL_divergence.detach(), self.last_KL_divergence, self.beta)
                KL_divergence = torch.clamp(KL_divergence, min=self.free_bits)
                self.last_KL_divergence = KL_divergence.detach()

            loss = self.beta[-1] * KL_divergence + recon_loss
            return loss
        else:
            mu_z_x, mu_z_y, logvar_z_x, logvar_z_y, z_x, stable, trend, recon = self.model(batch_data, batch_time, 0)

            if self.dataset == 'SMAP':
                batch_data[:, :, 1:] = torch.sigmoid(batch_data[:, :, 1:])

            contributions = kl_divergence_contributions(recon, batch_data)

            return recon, contributions

    def train(self):

        self.min_step = 1.0 / (self.epochs * len(self.train_loader))
        self.beta.append(self.min_step)

        for e in range(self.epochs):
            start = time()
            self.model.train()
            train_loss = []
            for batch in tqdm(self.train_loader):
                self.optimizer.zero_grad()
                loss = self._process_one_batch(batch, train=True, valid=False)
                train_loss.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            with torch.no_grad():
                self.model.eval()
                valid_loss = []
                for batch in tqdm(self.valid_loader):
                    loss = self._process_one_batch(batch, train=True, valid=True)
                    valid_loss.append(loss.item())

            train_loss, valid_loss = np.average(train_loss), np.average(valid_loss)
            end = time()
            self.T -= 1
            print(f'Epoch: {e} || Train Loss: {train_loss:.6f} Valid Loss: {valid_loss:.6f} || Cost: {end - start:.4f}')

            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                break

        self.model.load_state_dict(torch.load(self.model_dir + self.dataset + '_model.pkl'))

    def test(self):

        self.model.load_state_dict(torch.load(self.model_dir + self.dataset + '_model.pkl'))

        with torch.no_grad():
            self.model.eval()

            init_src, init_rec, init_KL_contribute = [], [], []
            for batch in tqdm(self.train_loader):
                recon, contributions = self._process_one_batch(batch, train=False, valid=False)
                batch_data = batch['data']

                init_src.append(np.concatenate(batch_data.detach().cpu().numpy()[:, -self.step_size:, :], axis=0))
                init_rec.append(np.concatenate(recon.detach().cpu().numpy()[:, -self.step_size:, :], axis=0))
                init_KL_contribute.append(
                    np.concatenate(contributions.detach().cpu().numpy()[:, -self.step_size:, :], axis=0))

            test_label, test_src, test_rec, test_KL_contribute = [], [], [], []
            for batch in tqdm(self.test_loader):
                recon, contributions = self._process_one_batch(batch, train=False, valid=False)
                batch_data = batch['data']
                batch_label = batch['label']

                test_label.append(np.concatenate(batch_label.detach().cpu().numpy()[:, -self.step_size:, :], axis=0))
                test_src.append(np.concatenate(batch_data.detach().cpu().numpy()[:, -self.step_size:, :], axis=0))
                test_rec.append(np.concatenate(recon.detach().cpu().numpy()[:, -self.step_size:, :], axis=0))
                test_KL_contribute.append(
                    np.concatenate(contributions.detach().cpu().numpy()[:, -self.step_size:, :], axis=0))

        # init-score
        init_src = np.concatenate(init_src, axis=0)
        init_rec = np.concatenate(init_rec, axis=0)
        init_KL_contribute = np.concatenate(init_KL_contribute, axis=0)

        if self.dataset == 'MSL' or self.dataset == 'SMAP':
            condition = test_rec[:, 1:] > 0.5
            init_rec[:, 1:] = np.where(condition, 1.0, 0.0)

        init_mse = ((init_src - init_rec) ** 2) * np.abs(init_KL_contribute)
        init_score = np.mean(init_mse, axis=-1, keepdims=True)

        # test-socre
        test_label = np.concatenate(test_label, axis=0)
        test_src = np.concatenate(test_src, axis=0)
        test_rec = np.concatenate(test_rec, axis=0)
        test_KL_contribute = np.concatenate(test_KL_contribute, axis=0)

        if self.dataset == 'MSL' or self.dataset == 'SMAP':
            condition = test_rec[:, 1:] > 0.5
            test_rec[:, 1:] = np.where(condition, 1.0, 0.0)

        test_mse = ((test_src - test_rec) ** 2) * np.abs(test_KL_contribute)
        test_score = np.mean(test_mse, axis=-1, keepdims=True)

        res = evaluate(init_score.reshape(-1), test_score.reshape(-1), test_label.reshape(-1), self.anomaly_ratio)
        print("\n=============== " + self.dataset + " ===============")
        print(f"P: {res['precision']:.4f} || R: {res['recall']:.4f} || F1: {res['f1_score']:.4f}")
        print("=============== " + self.dataset + " ===============\n")

        print("\n=============== " + self.dataset + " ===============")
        print(f"AUC: {res['auc']:.4f}")
        print("=============== " + self.dataset + " ===============\n")

        today = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

        if not os.path.exists(f'{self.model_dir}/{today}/'):
            os.makedirs(f'{self.model_dir}/{today}/')

        self.all_res += f"threshold\n{res['threshold']}\n"
        self.all_res += f"precision\n{res['precision']}\n"
        self.all_res += f"recall\n{res['recall']}\n"
        self.all_res += f"f1_score\n{res['f1_score']}\n"

        with open(f'{self.model_dir}/{today}/result.txt', 'a') as file:
            file.write(self.all_res + '\n')

        return res['f1_score'], res['precision'], res['recall']