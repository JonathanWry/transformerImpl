import torch
from torch import nn
import math

from model.embedding.PositionalEncoding import PositionalEncoding
from model.model.Decoder import Decoder
from model.model.Encoder import Encoder


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, emb_dim, nhead, num_encoder_layers, num_decoder_layers,
                 dim_feedforward, drop_prob, max_len):
        super(Transformer, self).__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, emb_dim)
        self.trg_embedding = nn.Embedding(trg_vocab_size, emb_dim)
        self.positional_encoding = PositionalEncoding(emb_dim, max_len)
        self.encoder = Encoder(enc_vocab_size=trg_vocab_size,
                               emb_dim=emb_dim,
                               ffn_hidden=dim_feedforward,
                               n_layer=num_encoder_layers,
                               n_head=nhead,
                               drop_prob=drop_prob)
        self.decoder = Decoder(dec_vocab_size=trg_vocab_size,
                               emb_dim=emb_dim,
                               ffn_hidden=dim_feedforward,
                               n_layer=num_decoder_layers,
                               n_head=nhead,
                               drop_prob=drop_prob)
        self.fc_out = nn.Linear(emb_dim, trg_vocab_size)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, src, trg, padding_idx):
        batch_size, src_len = src.size()
        _, trg_len = trg.size()
        # Generate masks if not provided
        trg_mask = self.make_trg_mask(trg, padding_idx).to(trg.device)
        src_mask = self.make_src_mask(src, padding_idx).to(trg.device)
        src = self.src_embedding(src) * math.sqrt(self.src_embedding.embedding_dim)
        src = self.positional_encoding(src)
        trg = self.trg_embedding(trg) * math.sqrt(self.trg_embedding.embedding_dim)
        trg = self.positional_encoding(trg)
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(trg, encoder_output, trg_mask, src_mask)
        output = self.fc_out(decoder_output)
        return output

    def make_src_mask(self, src, padding_idx):
        src_pad_mask = (src != padding_idx).unsqueeze(1).unsqueeze(2)
        return src_pad_mask

    def make_trg_mask(self, trg, padding_idx):
        trg_seq_length = trg.size(1)
        pad_mask = (trg != padding_idx).unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, seq_len)
        # Causal mask
        causal_mask = torch.tril(torch.ones(trg_seq_length, trg_seq_length, device=trg.device)).type(torch.bool)
        # Combine padding and causal masks
        trg_mask = pad_mask & causal_mask.unsqueeze(0)
        return trg_mask

