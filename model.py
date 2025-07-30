import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import Parameter
from GlobalAttention import GlobalAttentionConv
from GraphTrans import GraphTransConv
import scipy.sparse as sp

device = f'cuda' if torch.cuda.is_available() else 'cpu'


class HGSAN(nn.Module):
    def __init__(self, num_edge, num_channels, w_in, w_out, num_class, norm, args=None):
        super(HGSAN, self).__init__()
        # 把定义的数值传过来
        self.num_edge = num_edge
        self.num_channels = num_channels
        self.w_in = w_in
        self.w_out = w_out
        self.num_class = num_class
        self.adj_norm = norm
        self.args = args
        self.conv = Conv(in_channels=num_edge, out_channels=1)
        self.GlobalAttention = GlobalAttentionConv(d_model=self.w_in, nhead=1, dim_feedforward=self.w_out)
        self.GraphTrans = GraphTransConv(in_channels=self.w_in * 2, hidden_channels=self.w_out, norm=self.adj_norm)
        self.loss = nn.CrossEntropyLoss()
        self.register_buffer("epsilon", torch.FloatTensor([1e-12]))

        self.linear1 = nn.Linear(self.w_out, self.w_out)
        self.linear2 = nn.Linear(self.w_out, self.num_class)

    def forward(self, A, X, target_x, target):
        # Global Self-Attention
        Original_Features = X
        GlobalAttention_z = self.GlobalAttention(Original_Features)
        X_G = torch.cat((GlobalAttention_z, X), dim=1)

        # Edges Conv
        A = A.unsqueeze(0).permute(0, 3, 1, 2)
        A = self.conv(A)
        A = torch.squeeze(A).to(device)
        W = (self.conv.weight).detach()

        # Graph Self-Attention
        Z = F.relu(self.GraphTrans(X_G, A, self.adj_norm))

        Z = self.linear1(Z)
        Z = F.relu(Z)
        y = self.linear2(Z[target_x])
        loss = self.loss(y, target)
        return loss, y, W


# Conv卷积
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv, self).__init__()
        self.in_channels = in_channels  # >>>5
        self.out_channels = out_channels  # >>>1
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, 1, 1))
        self.bias = None
        self.scale = nn.Parameter(torch.Tensor([0.1]), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        nn.init.constant_(self.weight, 0.2)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, A):
        A = torch.sum(A * F.softmax(self.weight, dim=1), dim=1)
        return A
