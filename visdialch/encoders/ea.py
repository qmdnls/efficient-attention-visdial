import torch
from torch import nn
from torch.nn import functional as F

class UtilityBlock(nn.Module):
    def __init__(self, hidden_dim, feedforward_dim=2048, nheads=8, dropout=0.1):
        super(UtilityBlock, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, nheads) # dropout?
        self.linear1 = nn.Linear(3*hidden_dim, feedforward_dim)
        self.linear2 = nn.Linear(feedforward_dim, hidden_dim)
        self.relu1 = nn.ReLU(feedforward_dim)
        self.relu2 = nn.ReLU(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm([hidden_dim], elementwise_affine=False)

    def forward(self, target, source_a, source_b):
        # Apply multihead attention mechanism for target and multiple sources as described in the paper
        out_t, _ = self.multihead_attn(target, target, target) # self attention for target utility
        out_a, _ = self.multihead_attn(target, source_a, source_a) # attention to source utility a
        out_b, _ = self.multihead_attn(target, source_b, source_b) # attention to source utility b
        out = torch.cat((out_t, out_a, out_b), dim=2) # concatenate the resulting output tensors
        out = self.dropout(out)
        out = self.relu1(self.linear1(out)) 
        out = self.dropout(out)
        out = self.relu2(self.linear2(out))
        out = out + target
        out = self.norm(out + target) # add & norm (residual target)
        return out

class UtilityLayer(nn.Module):
    def __init__(self, hidden_dim, feedforward_dim=2048, nheads=8, dropout=0.1):
        super(UtilityLayer, self).__init__()
        self.utility_v = UtilityBlock(hidden_dim, feedforward_dim, nheads, dropout)
        self.utility_q = UtilityBlock(hidden_dim, feedforward_dim, nheads, dropout)
        self.utility_r = UtilityBlock(hidden_dim, feedforward_dim, nheads, dropout)

    def forward(self, V, Q, R):
        V_out = self.utility_v(V, Q, R)
        Q_out = self.utility_q(Q, V, R)
        R_out = self.utility_r(R, V, Q)
        return V_out, Q_out, R_out

# Test below

utility = UtilityBlock(hidden_dim=10, feedforward_dim=2048, nheads=2, dropout=0.1)

v = torch.tensor([[[0.3, 0.4, 0.5, 0.6, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
                 [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
                 [0.5, 0.4, 0.3, 0.2, 0.1, 0.9, 0.8, 0.7, 0.6, 0.5]]])
q = torch.tensor([[[0.3, 0.2, 0.5, 0.6, 0.3, 0.16, 0.5, 0.42, 0.3, 0.2],
                 [0.1, 0.2, 0.7, 0.4, 0.7, 0.9, 0.79, 0.81, 0.9, 0.95],
                 [0.4, 0.1, 0.3, 0.4, 0.4, 0.9, 0.8, 0.7, 0.5, 0.5]]])
r = torch.tensor([[[0.3, 0.4, 0.5, 0.6, 0.2, 0.2, 0.2, 0.4, 0.4, 0.2],
                 [0.1, 0.8, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.1, 0.8],
                 [0.5, 0.5, 0.3, 0.2, 0.1, 0.3, 0.8, 0.7, 0.6, 0.5]]])

print(v.shape, q.shape, r.shape)

v2 = utility(v, q, r)


print("Success")
