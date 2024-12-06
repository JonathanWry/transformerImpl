from torch import nn
import math
from Transformer.layer.LayerNorm import LayerNorm
from Transformer.layer.MultiHeadAttention import MultiHeadAttention
from Transformer.layer.PositionwiseFeedForward import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    def __init__(self, emb_dim, n_head, ffn_hidden, drop_prob):
        super(DecoderLayer, self).__init__()
        self.selfAttention = MultiHeadAttention(emb_dim, n_head)
        self.norm1 = LayerNorm(hidden_size=emb_dim)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.enc_dec_attention = MultiHeadAttention(emb_dim, n_head)
        self.norm2 = LayerNorm(hidden_size=emb_dim)
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.feed_forward = PositionwiseFeedForward(emb_dim=emb_dim, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNorm(hidden_size=emb_dim)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, x, encoder_output, decoder_mask, encoder_mask):
        # compute self attention
        _x = x
        x, _ = self.selfAttention(q=x, k=x, v=x, attn_mask=decoder_mask)
        x = self.dropout1(x)
        x = x + _x
        x = self.norm1(x)
        # attention with encoder
        if (encoder_output is not None):
            _x = x
            x, _ = self.enc_dec_attention(q=x, k=encoder_output, v=encoder_output, attn_mask=encoder_mask)
            x = self.dropout2(x)
            x = x + _x
            x = self.norm2(x)
        # feed forward
        _x = x
        x = self.feed_forward(x)
        x = self.dropout3(x)
        x = x + _x
        x = self.norm3(x)
        return x
