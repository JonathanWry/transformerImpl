from torch import nn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, emb_dim, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(emb_dim, hidden)
        self.linear2 = nn.Linear(hidden, emb_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
