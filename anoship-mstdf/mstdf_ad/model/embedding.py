import math
import torch
from torch import nn


class PositionEmbedding(nn.Module):
    def __init__(self, model_dim, max_len=5000):
        super(PositionEmbedding, self).__init__()
        pe = torch.zeros(max_len, model_dim)
        pe.require_grad = False

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        x = self.pe[:, : x.size(1), :]

        return self.norm(x)


class TimeEmbedding(nn.Module):
    def __init__(self, model_dim, time_num):
        super(TimeEmbedding, self).__init__()
        self.conv = nn.Conv1d(in_channels=time_num, out_channels=model_dim, kernel_size=(1,))
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="leaky_relu")

        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        x = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)

        return self.norm(x)


def get_dim(x, model_dim):
    powers = []
    current_power = 1
    while current_power <= model_dim:
        if current_power >= x:
            powers.append(current_power)
        current_power *= 2
    return powers


class TemporalEmbedding(nn.Module):
    def __init__(self, device, window_size, model_dim, feature_num):
        super(TemporalEmbedding, self).__init__()

        self.device = device
        self.dims = get_dim(feature_num, model_dim)

        self.feature_num = feature_num
        self.num_layers = 2
        self.num_directions = 1

        self.lstm = nn.LSTM(feature_num, feature_num, self.num_layers, batch_first=True)

        self.conv = nn.Conv1d(in_channels=feature_num, out_channels=self.dims[0], kernel_size=(1,))
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="leaky_relu")

        self.conv1 = nn.ModuleList()
        self.conv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        for i in range(1, len(self.dims)):
            self.add_conv_layer(self.dims[i-1], self.dims[i], (1,))
            self.add_conv_transpose_layer(window_size, window_size, (2,), 2)
            self.norm.append(nn.LayerNorm(self.dims[i]))

        self.activation = nn.GELU()

    def add_conv_layer(self, in_channels, out_channels, kernel_size):
        conv_layer = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        nn.init.kaiming_normal_(conv_layer.weight, mode="fan_in", nonlinearity="leaky_relu")
        self.conv1.append(conv_layer)

    def add_conv_transpose_layer(self, in_channels, out_channels, kernel_size, stride):
        conv_layer = nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        nn.init.kaiming_normal_(conv_layer.weight, mode="fan_in", nonlinearity="leaky_relu")
        self.conv2.append(conv_layer)

    def forward(self, x, flag=True):
        if flag:
            batch_size = x.shape[0]
            h_0 = torch.zeros(self.num_directions * self.num_layers, batch_size, self.feature_num).to(self.device)
            c_0 = torch.zeros(self.num_directions * self.num_layers, batch_size, self.feature_num).to(self.device)
            x, _ = self.lstm(x, (h_0, c_0))

        x = self.activation(self.conv(x.permute(0, 2, 1))).permute(0, 2, 1)

        for conv1, conv2, norm in zip(self.conv1, self.conv2, self.norm):
            x1 = self.activation(conv1(x.permute(0, 2, 1)).permute(0, 2, 1))
            x2 = self.activation(conv2(x))
            x = norm(x1 + x2)

        return x

class SpatialEmbedding(nn.Module):
    def __init__(self, device, window_size, model_dim, feature_num):
        super(SpatialEmbedding, self).__init__()

        self.device = device
        self.dims = get_dim(window_size, model_dim)

        self.feature_num = feature_num
        self.num_layers = 2
        self.num_directions = 1

        self.lstm = nn.LSTM(feature_num, feature_num, self.num_layers, batch_first=True)

        self.conv = nn.Conv1d(in_channels=window_size, out_channels=self.dims[0], kernel_size=(1,))
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="leaky_relu")

        self.conv1 = nn.ModuleList()
        self.conv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        for i in range(1, len(self.dims)):
            self.add_conv_layer(self.dims[i - 1], self.dims[i], (1,))
            self.add_conv_transpose_layer(feature_num, feature_num, (2,), 2)
            self.norm.append(nn.LayerNorm(feature_num))

        self.activation = nn.GELU()

    def add_conv_layer(self, in_channels, out_channels, kernel_size):
        conv_layer = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        nn.init.kaiming_normal_(conv_layer.weight, mode="fan_in", nonlinearity="leaky_relu")
        self.conv1.append(conv_layer)

    def add_conv_transpose_layer(self, in_channels, out_channels, kernel_size, stride):
        conv_layer = nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride)
        nn.init.kaiming_normal_(conv_layer.weight, mode="fan_in", nonlinearity="leaky_relu")
        self.conv2.append(conv_layer)

    def forward(self, x, flag=True):
        if flag:
            batch_size = x.shape[0]
            h_0 = torch.zeros(self.num_directions * self.num_layers, batch_size, self.feature_num).to(self.device)
            c_0 = torch.zeros(self.num_directions * self.num_layers, batch_size, self.feature_num).to(self.device)
            x, _ = self.lstm(x, (h_0, c_0))

        x = self.activation(self.conv(x))

        for conv1, conv2, norm in zip(self.conv1, self.conv2, self.norm):
            x1 = self.activation(conv1(x))
            x2 = self.activation(conv2(x.permute(0, 2, 1))).permute(0, 2, 1)
            x = norm(x1 + x2)
        x = x.permute(0, 2, 1)
        return x