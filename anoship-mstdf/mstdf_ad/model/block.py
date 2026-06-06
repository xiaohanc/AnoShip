import torch
import torch.nn.functional as F
from torch import nn

from mstdf_ad.model.attention import Attention


class TransformerBlock(nn.Module):
    def __init__(self, model_dim, ff_dim, atten_dim, head_num, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = Attention(model_dim, atten_dim, head_num, dropout)

        self.conv1 = nn.Conv1d(in_channels=model_dim, out_channels=ff_dim, kernel_size=(1,))
        self.conv2 = nn.Conv1d(in_channels=ff_dim, out_channels=model_dim, kernel_size=(1,))
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="leaky_relu")

        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, q, k, v):
        residual = v.clone()
        x = self.attention(q, k, v)
        x = self.norm1(x + residual)

        x = self.activation(self.conv1(x.permute(0, 2, 1)))
        x = self.dropout(self.activation(self.conv2(x)).permute(0, 2, 1))

        return self.norm2(x + residual)


class SpatialTemporalTransformerBlock(nn.Module):
    def __init__(self, feature_num, window_size, model_dim, ff_dim, atten_dim, head_num, dropout):
        super(SpatialTemporalTransformerBlock, self).__init__()

        self.temporal_attention = TransformerBlock(model_dim, ff_dim, atten_dim, head_num, dropout)
        self.spatial_attention = TransformerBlock(model_dim, ff_dim, atten_dim, head_num, dropout)
        self.ordinary_attention = TransformerBlock(model_dim, ff_dim, atten_dim, head_num, dropout)

        self.conv1 = nn.Conv1d(in_channels=feature_num, out_channels=model_dim, kernel_size=(1,))
        self.conv2 = nn.Conv1d(in_channels=model_dim, out_channels=window_size, kernel_size=(1,))
        self.conv3 = nn.Conv1d(in_channels=2 * model_dim, out_channels=model_dim, kernel_size=(1,))
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv3.weight, mode="fan_in", nonlinearity="leaky_relu")

        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

        self.norm = nn.LayerNorm(model_dim)

    def forward(self, xt, xs):
        residual = xt.clone()
        xt = self.temporal_attention(xt, xt, xt)
        xs = self.spatial_attention(xs, xs, xs)

        x = self.activation(self.conv1(xs))
        x = self.activation(self.conv2(x.permute(0, 2, 1)))
        x = torch.cat((xt, x), dim=-1)

        x = self.activation(self.conv3(x.permute(0, 2, 1)))
        x = self.dropout(self.activation(x).permute(0, 2, 1))
        x = self.ordinary_attention(x, x, x)

        return self.norm(x + residual), xs

