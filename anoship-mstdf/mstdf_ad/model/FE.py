import torch
import torch.nn as nn
import torch.nn.functional as F

class FE(nn.Module):
    def __init__(self, window_size, feature_num, kernel_size, stride):
        super(FE, self).__init__()

        self.kernel_size=kernel_size
        self.stride=stride

        self.conv1 = nn.Conv1d(in_channels=window_size, out_channels=window_size, kernel_size=(1,))
        self.conv2 = nn.Conv1d(in_channels=2+self.kernel_size, out_channels=window_size, kernel_size=(1,))
        self.conv3 = nn.Conv1d(in_channels=(window_size - self.kernel_size) // self.stride + 1, out_channels=1, kernel_size=(1,))
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv3.weight, mode="fan_in", nonlinearity="leaky_relu")

        self.emb_global = nn.Sequential(
            self.conv1,
            nn.GELU(),
        )

        self.emb_local = nn.Sequential(
            self.conv2,
            nn.GELU(),
        )

        self.out_linear = nn.Sequential(
            self.conv3,
            nn.GELU()
        )

    def get_conditon(self, x):
        x_g = x
        f_global = torch.fft.rfft(x_g[:, :, :-1], dim=-1)
        f_global = torch.cat((f_global.real, f_global.imag), dim=-1)
        f_global = self.emb_global(f_global.permute(0, 2, 1)).permute(0, 2, 1)
        x_g = x_g.reshape(x.shape[0], x.shape[1], 1, -1)
        x_l = x_g.clone()
        unfold = nn.Unfold(
            kernel_size=(1, self.kernel_size),
            dilation=1,
            padding=0,
            stride=(1, self.stride),
        )
        unfold_x = unfold(x_l)
        unfold_x = unfold_x.reshape(x.shape[0] * x.shape[1], self.kernel_size, -1)
        unfold_x = unfold_x.permute(0, 2, 1)
        f_local = torch.fft.rfft(unfold_x, dim=-1)
        f_local = torch.cat((f_local.real, f_local.imag), dim=-1)
        f_local = self.emb_local(f_local.permute(0, 2, 1)).permute(0, 2, 1)
        f_local = self.out_linear(f_local)
        f_local = f_local.reshape(x.shape[0], x.shape[1], -1)
        output = torch.cat((f_global.permute(0, 2, 1), f_local.permute(0, 2, 1)), -1)
        return output

    def forward(self, x, y):
        f_x = x.clone()
        f_y = y.clone()
        f_x = self.get_conditon(f_x.permute(0, 2, 1))
        f_y = self.get_conditon(f_y.permute(0, 2, 1))

        f_x = torch.cat((x, f_x), dim=-1)
        f_y = torch.cat((y, f_y), dim=-1)

        return f_x, f_y
