from torch import nn

from model.block.DecoderLayer import DecoderLayer


class Decoder(nn.Module):
    def __init__(self, dec_vocab_size, emb_dim, ffn_hidden, n_layer, n_head, drop_prob):
        super(Decoder, self).__init__()
        self.Layers = nn.ModuleList([DecoderLayer(emb_dim=emb_dim,
                                                  n_head=n_head,
                                                  ffn_hidden=ffn_hidden,
                                                  drop_prob=drop_prob) for _ in
                                     range(n_layer)])

    def forward(self, x, encoder_output, decoder_mask, encoder_mask):
        for layer in self.Layers:
            x = layer(x, encoder_output, decoder_mask, encoder_mask)
        return x
