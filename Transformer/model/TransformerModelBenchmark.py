import math

import torch
from torch import nn

from Transformer.embedding.PositionalEncoding import PositionalEncoding


class TransformerModelBenchmark(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, emb_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, max_len):
        super(TransformerModelBenchmark, self).__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, emb_dim)
        self.trg_embedding = nn.Embedding(trg_vocab_size, emb_dim)
        self.positional_encoding = PositionalEncoding(emb_dim, max_len)

        self.transformer = nn.Transformer(
            d_model=emb_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.fc_out = nn.Linear(emb_dim, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, trg, padding_idx, src_padding_mask=None, trg_padding_mask=None):
      batch_size, src_len = src.size()
      _, trg_len = trg.size()

      # Generate masks if not provided
      trg_mask = self.generate_square_subsequent_mask(trg_len).to(trg.device)

      if src_padding_mask is None:
          # Use raw 2D `src` tensor to generate padding mask
          src_padding_mask = self.generate_padding_mask(src, padding_idx)

      if trg_padding_mask is None:
          # Use raw 2D `trg` tensor to generate padding mask
          trg_padding_mask = self.generate_padding_mask(trg, padding_idx)

      # Embedding and positional encoding for source
      src = self.src_embedding(src) * math.sqrt(self.src_embedding.embedding_dim)
      src = self.positional_encoding(src)

      # Embedding and positional encoding for target
      trg = self.trg_embedding(trg) * math.sqrt(self.trg_embedding.embedding_dim)
      trg = self.positional_encoding(trg)
      # print("src_padding_mask:", src_padding_mask.shape)
      # print("trg_padding_mask:", trg_padding_mask.shape)
      # print("trg_mask:", trg_mask.shape)


      # Transformer forward pass
      output = self.transformer(
          src.transpose(0, 1),  # Shape: (seq_len, batch_size, emb_dim)
          trg.transpose(0, 1),  # Shape: (seq_len, batch_size, emb_dim)
          src_mask=None,
          tgt_mask=trg_mask,  # Shape: (trg_len, trg_len)
          src_key_padding_mask=src_padding_mask,  # Shape: (batch_size, src_len)
          tgt_key_padding_mask=trg_padding_mask   # Shape: (batch_size, trg_len)
      )

      # Output layer
      output = self.fc_out(output.transpose(0, 1))  # Convert back to shape (batch_size, seq_len, trg_vocab_size)

      return output

    def generate_square_subsequent_mask(self, size):
        """
        Generate a square mask for the sequence to mask future tokens (causal mask).
        """
        mask = torch.triu(torch.full((size, size), float('-inf')), diagonal=1)
        return mask

    def generate_padding_mask(self, sequences, padding_idx):
        return sequences == padding_idx


