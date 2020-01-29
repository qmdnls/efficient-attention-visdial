import torch
from torch import nn
from torch.nn import functional as F

class UtilityBlock(nn.Module):
    """Efficient attention mechanism for many utilities block implemented for the visual dialog task (here: three utilities).

    Args:
        hidden_dim: dimension of the feature vector. Also the dimension of the final context vector provided to the decoder (required).
        feedforward_dim: dimension of the hidden feedforward layer, implementation details from "Attention is all you need" (default=2048).
        nhead: the number of heads in the multihead attention layers (default=8).
        dropout: the dropout probability (default=0.1).
    """
    def __init__(self, hidden_dim, feedforward_dim=2048, nhead=8, dropout=0.1):
        super(UtilityBlock, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, nhead) # dropout?
        self.linear1 = nn.Linear(3*hidden_dim, feedforward_dim)
        self.linear2 = nn.Linear(feedforward_dim, hidden_dim)
        self.relu1 = nn.ReLU(feedforward_dim)
        self.relu2 = nn.ReLU(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm([hidden_dim], elementwise_affine=False)

    def forward(self, target, source_a, source_b):
        """Passes the inputs through the utility attention block. For a detailed description see the paper. Inputs are tensors for each utility. The output is the updated utility tensor.

        Args:
            target: the target utility. The output will be of the same shape as this target utility.
            source_a: the first source utility to attend to.
            source_b: the second source utility to attend to.
        """
        # Apply multihead attention mechanism for target and multiple sources as described in the paper
        out_t, _ = self.multihead_attn(target, target, target) # self attention for target utility
        out_a, _ = self.multihead_attn(target, source_a, source_a) # attention to source utility a
        out_b, _ = self.multihead_attn(target, source_b, source_b) # attention to source utility b
        out = torch.cat((out_t, out_a, out_b), dim=2) # concatenate the resulting output tensors
        out = self.dropout(out)
        out = self.relu1(self.linear1(out)) 
        out = self.dropout(out)
        out = self.relu2(self.linear2(out))
        out = self.norm(out + target) # add & norm (residual target)
        return out

class UtilityLayer(nn.Module):
    """Efficient attention mechanism for many utilities layer implemented for the visual dialog task (here: three utilities). The layer consist of three parallel utility attention blocks.

    Args:
        hidden_dim: dimension of the feature vector. Also the dimension of the final context vector provided to the decoder (required).
        feedforward_dim: dimension of the hidden feedforward layer, implementation details from "Attention is all you need" (default=2048).
        nhead: the number of heads in the multihead attention layers (default=8).
        dropout: the dropout probability (default=0.1).
    """
    def __init__(self, hidden_dim, feedforward_dim=2048, nhead=8, dropout=0.1):
        super(UtilityLayer, self).__init__()
        self.utility_v = UtilityBlock(hidden_dim, feedforward_dim, nhead, dropout)
        self.utility_q = UtilityBlock(hidden_dim, feedforward_dim, nhead, dropout)
        self.utility_r = UtilityBlock(hidden_dim, feedforward_dim, nhead, dropout)

    def forward(self, V, Q, R):
        """Passes the input utilities through the utility attention layer. Inputs are passed through their respective blocks in parallel. The output are the three updated utility tensors.

        Args:
            V: the visual utility tensor
            Q: the question utility tensor
            R: the history utility tensor
        """
        V_out = self.utility_v(V, Q, R)
        Q_out = self.utility_q(Q, V, R)
        R_out = self.utility_r(R, V, Q)
        return V_out, Q_out, R_out

# Test below

utility = UtilityBlock(hidden_dim=10, feedforward_dim=2048, nhead=2, dropout=0.0)

v = torch.tensor([[[0.3, 0.4, 0.5, 0.6, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
                 [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
                 [0.5, 0.4, 0.3, 0.2, 0.1, 0.9, 0.8, 0.7, 0.6, 0.5]]])
q = torch.tensor([[[0.3, 0.2, 0.5, 0.6, 0.3, 0.16, 0.5, 0.42, 0.3, 0.2],
                 [0.1, 0.2, 0.7, 0.4, 0.7, 0.9, 0.79, 0.81, 0.9, 0.95],
                 [0.4, 0.1, 0.3, 0.4, 0.4, 0.9, 0.8, 0.7, 0.5, 0.5]]])
r = torch.tensor([[[0.3, 0.4, 0.5, 0.6, 0.2, 0.2, 0.2, 0.4, 0.4, 0.2],
                 [0.1, 0.8, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.1, 0.8],
                 [0.5, 0.5, 0.3, 0.2, 0.1, 0.3, 0.8, 0.7, 0.6, 0.5]]])

print("Input shapes:", v.shape, q.shape, r.shape)

v2 = utility(v, q, r)
q2 = utility(q, v, r)
r2 = utility(r, v, q)

print("Utility output shapes:", v2.shape, q2.shape, r2.shape)
print("Utility output:")
print(v2)
print(q2)
print(r2)


layer = UtilityLayer(hidden_dim=10, feedforward_dim=2048, nhead=2, dropout=0.0)

v_out, q_out, r_out = layer(v, q, r)

print("Layer output:")
print(v_out)
print(q_out)
print(r_out)
