import os.path as osp 
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import pickle
import scipy.sparse as sp

from torch_geometric.nn import TransformerConv

# from transformer_conv import TransformerConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GraphTransConv(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, norm, heads=1, dropout=0.5):
        super().__init__()
        self.convs1 = TransformerConv(hidden_channels, hidden_channels, heads,
                                      concat=False, beta=True, dropout=dropout)
        self.convs2 = TransformerConv(hidden_channels, hidden_channels, heads,
                                      concat=False, beta=True, dropout=dropout)
        self.norms1 = torch.nn.LayerNorm(hidden_channels)
        self.norms2 = torch.nn.LayerNorm(hidden_channels)
        self.adj_norm = norm
        self.gc1 = GraphConv(input=in_channels, output=hidden_channels, norm=self.adj_norm)
        self.gc2 = GraphConv(input=hidden_channels, output=hidden_channels, norm=self.adj_norm)

    def forward(self, x, adj, norm):
        H = adj
        # 把邻接矩阵转为稀疏形式
        adj = adj.cpu().detach().numpy()
        adj = sp.coo_matrix(adj)
        values = adj.data  # 边上对应权重值weight
        indices = np.vstack((adj.row, adj.col))  # pyG真正需要的coo形式
        edge_index = torch.cuda.LongTensor(indices)  # 我们真正需要的coo形式

        x = F.relu(self.gc1(x, H, norm))
        x = F.dropout(x, p=0.5, training=self.training)  # ACM=0.5  DBLP=0.0   IMDB=0.5
        x = self.norms1(self.convs1(x, edge_index)).relu()

        x = F.relu(self.gc2(x, H, norm))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.norms2(self.convs2(x, edge_index)).relu()
        return x


class GraphConv(nn.Module):
    def __init__(self, input, output, norm):
        super(GraphConv, self).__init__()
        self.in_channels = input
        self.out_channels = output
        self.adj_norm = norm
        self.weight = nn.Parameter(torch.Tensor(input, output))
        self.bias = nn.Parameter(torch.Tensor(output))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, X, adj,norm):
        X = torch.mm(X, self.weight)
        # H = self.norm(adj, add=True)
        if self.adj_norm == 'true':
            H = self.norm(adj, add=True)
            H=H.t()
        else:
            H = adj
        return torch.mm(H, X)

    def norm(self, H, add=False):
        H = H.t()
        if add == False:
            H = H * ((torch.eye(H.shape[0]) == 0).type(torch.cuda.FloatTensor))
        else:
            H = H * ((torch.eye(H.shape[0]) == 0).type(torch.cuda.FloatTensor)) + torch.eye(H.shape[0]).type(
                torch.cuda.FloatTensor)
        deg = torch.sum(H, dim=1)
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        deg_inv = deg_inv * torch.eye(H.shape[0]).type(torch.cuda.FloatTensor)
        H = torch.mm(deg_inv, H)
        H = H.t()
        return H
