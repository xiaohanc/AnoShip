import torch
from torch import nn

import torch.nn.functional as F


class GetTrend(nn.Module):
    def __init__(self, device, window_size, feature_num, model_dim, hidden_dim, ff_dim):
        super(GetTrend, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=model_dim, out_channels=ff_dim, kernel_size=(1,))
        self.conv2 = nn.Conv1d(in_channels=ff_dim, out_channels=hidden_dim, kernel_size=(1,))
        self.conv3 = nn.Conv1d(in_channels=hidden_dim, out_channels=ff_dim, kernel_size=(1,))
        self.conv4 = nn.Conv1d(in_channels=ff_dim, out_channels=feature_num, kernel_size=(1,))
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv3.weight, mode="fan_in", nonlinearity="leaky_relu")

        self.ae = nn.Sequential(
            self.conv1,
            nn.GELU(),
            self.conv2,
            nn.GELU(),
            self.conv3,
            nn.GELU(),
            self.conv4,
            nn.GELU(),
        )

    def forward(self, data, xt):
        x = data - xt
        x = self.ae(x.permute(0, 2, 1)).permute(0, 2, 1)

        return x