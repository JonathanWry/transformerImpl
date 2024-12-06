from torch import nn

from model.block.EncoderLayer import EncoderLayer


class Encoder(nn.Module):
    def __init__(self, enc_vocab_size, emb_dim, ffn_hidden, n_layer, n_head, drop_prob):
        super(Encoder, self).__init__()
        self.Layers = nn.ModuleList([EncoderLayer(emb_dim=emb_dim, ffn_hidden=ffn_hidden, n_head=n_head,
                                                  drop_prob=drop_prob) for _ in
                                     range(n_layer)])


    def forward(self, x, mask):
        for layer in self.Layers:
            x = layer(x, mask)
        return x
