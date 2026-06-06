import numpy as np
import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, model_dim, atten_dim, head_num, dropout):
        super(Attention, self).__init__()
        self.atten_dim = atten_dim
        self.head_num = head_num

        self.W_Q = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)
        self.W_K = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)
        self.W_V = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)

        self.fc = nn.Linear(self.atten_dim * self.head_num, model_dim, bias=True)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, Q, K, V):
        residual = V.clone()

        Q = self.W_Q(Q).view(Q.size(0), Q.size(1), self.head_num, self.atten_dim)
        K = self.W_K(K).view(K.size(0), K.size(1), self.head_num, self.atten_dim)
        V = self.W_V(V).view(V.size(0), V.size(1), self.head_num, self.atten_dim)

        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.atten_dim)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)

        context = context.transpose(1, 2)
        context = context.reshape(residual.size(0), residual.size(1), -1)
        output = self.dropout(self.fc(context))

        return self.norm(output)
