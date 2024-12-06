from torch import nn

from Transformer.layer.ScaleDotProductAttention import ScaleDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(emb_dim, emb_dim)
        self.w_k = nn.Linear(emb_dim, emb_dim)
        self.w_v = nn.Linear(emb_dim, emb_dim)
        self.w_o = nn.Linear(emb_dim, emb_dim)

    def forward(self, q, k, v, attn_mask):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)
        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=attn_mask)
        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_o(out)
        return out, attention

    def split(self, tensor):
        """
         split tensor by number of head

         :param tensor: [batch_size, length, emb_dim]
         :return: [batch_size, head, length, head_dim]
        """
        batch_size, length, emb_dim = tensor.size()
        dim_per_head = emb_dim // self.n_head
        assert emb_dim % self.n_head == 0, "Embedding dimension must be divisible by the number of heads."
        tensor = tensor.view(batch_size, length, self.n_head, dim_per_head).transpose(1, 2)
        return tensor

    def concat(self, tensor):
        """
          inverse function of self.split(tensor : torch.Tensor)

          :param tensor: [batch_size, head, length, head_dim]
          :return: [batch_size, length, emb_dim]
        """
        batch_size, head, length, head_dim = tensor.size()
        emb_dim = head * head_dim
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, emb_dim)
        return tensor
