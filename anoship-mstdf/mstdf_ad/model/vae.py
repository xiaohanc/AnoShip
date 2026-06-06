import torch
from torch import nn
import torch.nn.functional as F
from mstdf_ad.model.embedding import PositionEmbedding, TimeEmbedding, SpatialEmbedding, TemporalEmbedding
from mstdf_ad.model.block import SpatialTemporalTransformerBlock

class Embedding(nn.Module):
    def __init__(self, device, window_size, feature_num, model_dim, time_num):
        super(Embedding, self).__init__()

        self.data_t_embedding = TemporalEmbedding(device, window_size, model_dim, feature_num)
        self.data_s_embedding = SpatialEmbedding(device, window_size, model_dim, feature_num)
        self.time_t_embedding = TimeEmbedding(model_dim, time_num)
        self.position_t_embedding = PositionEmbedding(model_dim)
        self.position_s_embedding = PositionEmbedding(model_dim)

    def forward(self, x, time):

        ori_x = self.data_t_embedding(x) + self.time_t_embedding(time)
        xt = ori_x + self.position_t_embedding(x)
        xs = self.data_s_embedding(x) + self.position_s_embedding(x.permute(0, 2, 1))

        return ori_x, xt, xs

class Encoder(nn.Module):
    def __init__(self, device, window_size, feature_num, model_dim, ff_dim, atten_dim, hidden_dim, block_num, head_num, time_num, dropout, epsilon=1e-4):
        super(Encoder, self).__init__()

        self.epsilon = epsilon

        self.encoder_blocks = nn.ModuleList()
        for i in range(block_num):
            dp = 0 if i == block_num - 1 else dropout
            residual = True if i == block_num - 1 else False
            self.encoder_blocks.append(
                    SpatialTemporalTransformerBlock(feature_num, window_size, model_dim, ff_dim, atten_dim, head_num, dp)
            )

        self.conv1 = nn.Conv1d(in_channels=model_dim, out_channels=ff_dim, kernel_size=(1,))
        self.conv2 = nn.Conv1d(in_channels=ff_dim, out_channels=model_dim, kernel_size=(1,))
        self.conv3 = nn.Conv1d(in_channels=model_dim, out_channels=hidden_dim, kernel_size=(1,))
        self.conv4 = nn.Conv1d(in_channels=model_dim, out_channels=hidden_dim, kernel_size=(1,))
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv3.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv4.weight, mode="fan_in", nonlinearity="leaky_relu")
        self.activation = F.gelu

        self.fc_mu = nn.Sequential(
            self.conv3
        )

        self.fc_var = nn.Sequential(
            self.conv4,
            nn.Softplus(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 + logvar)
        eps = torch.randn_like(std)

        return eps * std + mu

    def encode(self, xt, xs):
        for encoder_block in self.encoder_blocks:
            xt, xs = encoder_block(xt, xs)

        x = xt.clone()
        x = self.activation(self.conv1(x.permute(0, 2, 1)))
        x = self.activation(self.conv2(x)).permute(0, 2, 1)

        mu = self.fc_mu(x.permute(0, 2, 1)).permute(0, 2, 1)
        logvar = self.fc_var(x.permute(0, 2, 1)).permute(0, 2, 1)

        return mu, logvar, xt

    def forward(self, xt, xs):

        mu, logvar, xt = self.encode(xt, xs)

        z = self.reparameterize(mu, logvar)

        return mu, logvar, z, xt

class Decoder(nn.Module):
    def __init__(self, device, window_size, feature_num, model_dim, ff_dim, hidden_dim):
        super(Decoder, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=hidden_dim, out_channels=ff_dim, kernel_size=(1,))
        self.conv2 = nn.Conv1d(in_channels=ff_dim, out_channels=model_dim, kernel_size=(1,))
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="leaky_relu")
        self.activation = F.gelu

        self.conv3 = nn.Conv1d(in_channels=model_dim, out_channels=feature_num, kernel_size=(1,))
        nn.init.kaiming_normal_(self.conv3.weight, mode="fan_in", nonlinearity="leaky_relu")

        self.fc_mu = nn.Sequential(
            self.conv3
        )

    def forward(self, z):

        z = self.activation(self.conv1(z.permute(0, 2, 1)))
        z = self.activation(self.conv2(z)).permute(0, 2, 1)
        mu = self.fc_mu(z.permute(0, 2, 1)).permute(0, 2, 1)

        return mu
