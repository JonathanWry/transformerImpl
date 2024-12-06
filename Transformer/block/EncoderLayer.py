import torch
from torch import nn
import math

from Transformer.layer.LayerNorm import LayerNorm
from Transformer.layer.MultiHeadAttention import MultiHeadAttention
from Transformer.layer.PositionwiseFeedForward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, emb_dim, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.selfAttention = MultiHeadAttention(emb_dim=emb_dim, n_head=n_head)
        self.norm1 = LayerNorm(emb_dim)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.feed_forward = PositionwiseFeedForward(emb_dim=emb_dim, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(emb_dim)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, mask):
        _x = x
        x, _ = self.selfAttention(q=x, k=x, v=x, attn_mask=mask)
        x = self.dropout1(x)
        x = x + _x
        x =self.norm1(x)
        _x = x
        x = self.feed_forward(x)
        x = self.dropout2(x)
        x = x + _x
        x = self.norm2(x)
        return x
