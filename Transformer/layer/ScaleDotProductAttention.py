import torch
from torch import nn
import math


class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        # input is [batch_size, head, length, d_head]
        batch_size, head, length, d_head = k.size()
        k_t = k.transpose(-2, -1)
        score = (q @ k_t) / math.sqrt(d_head)
        #  print("in scaleDot: score shape", score.shape)
        #  print("in scaleDot: mask shape", mask.shape)
        if mask is not None:
            score = score.masked_fill(mask == 0, float('-inf'))
        score = self.softmax(score)
        v = score @ v
        return v, score
