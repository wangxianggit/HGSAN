import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalAttentionConv(nn.Module):

    def __init__(self, d_model, nhead=1, dim_feedforward=128):
        super(GlobalAttentionConv, self).__init__()
        self.EncoderLayer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                       dropout=0.5, activation='relu')
        self.TransformerEncoder = nn.TransformerEncoder(encoder_layer=self.EncoderLayer, num_layers=1)

    def forward(self, x, mask=None, src_key_padding_mask=None):
        x = F.dropout(x, p=0.5, training=self.training)  # IMDB=0.5  ACM=0.5  DBLP=0.0
        out = self.TransformerEncoder(x)
        return out
