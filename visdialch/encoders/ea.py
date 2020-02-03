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
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, nhead) # dropout? separate attention modules?
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

def sum_attention(nnet, query, value, mask=None, dropout=None, mode='1D'):
	if mode == '2D':
		batch, dim = query.size(0), query.size(1)
		query = query.permute(0, 2, 3, 1).view(batch, -1, dim)
		value = value.permute(0, 2, 3, 1).view(batch, -1, dim)
		mask = mask.view(batch, 1, -1)

	scores = nnet(query).transpose(-2, -1)
	if mask is not None:
		scores.data.masked_fill_(mask.eq(0), -65504.0)

	p_attn = F.softmax(scores, dim=-1)
	if dropout is not None:
		p_attn = dropout(p_attn)
	weighted = torch.matmul(p_attn, value)

	return weighted, p_attn

class SummaryAttn(nn.Module):
	def __init__(self, dim, num_attn, dropout, is_multi_head=False, mode='1D'):
		super(SummaryAttn, self).__init__()
		self.linear = nn.Sequential(
			nn.Linear(dim, dim),
			nn.ReLU(inplace=True),
			nn.Linear(dim, num_attn),
		)
		self.h = num_attn
		self.is_multi_head = is_multi_head
		self.attn = None
		self.dropout = nn.Dropout(p=dropout) if dropout else None
		self.mode = mode

	def forward(self, query, value, mask=None):
		if mask is not None:
			mask = mask.unsqueeze(1)
		batch = query.size(0)

		weighted, self.attn = sum_attention(self.linear, query, value, mask=mask, dropout=self.dropout, mode=self.mode)
		weighted = weighted.view(batch, -1) if self.is_multi_head else weighted.mean(dim=-2)

		return weighted

class UtilityEncoder(nn.Module):
    """Efficient attention mechanism for many utilities encoder implemented for the visual dialog task (Nguyen et al. 2019, https://arxiv.org/abs/1911.11390). Outputs an encoding that can be processed by the decoder module.

    Args:
        config: configuration dictionary
        vocabulary: 
    """
    def __init__(self, config, vocabulary):
        super().__init__()
        self.config = config

        self.word_embed = nn.Embedding(
            len(vocabulary),
            config["word_embedding_size"],
            padding_idx=vocabulary.PAD_INDEX
        )
        
        self.hist_rnn = nn.LSTM(
            config["word_embedding_size"],
            config["lstm_hidden_size"],
            config["lstm_enc_num_layers"],
            batch_first=True,
            dropout=config["lstm_dropout"],
            bidirectional=True
        )
        
        self.ques_rnn = nn.LSTM(
            config["word_embedding_size"],
            config["lstm_hidden_size"],
            config["lstm_enc_num_layers"],
            batch_first=True,
            dropout=config["lstm_dropout"],
            bidirectional=True
        )
        
        self.hist_rnn = DynamicRNN(self.hist_rnn)
        self.ques_rnn = DynamicRNN(self.ques_rnn)
       
        self.util1 = UtilityLayer(hidden_dim=config["lstm_hidden_size"], feedforward_dim=2048, nhead=8, dropout=0.1)
        self.util2 = UtilityLayer(hidden_dim=config["lstm_hidden_size"], feedforward_dim=2048, nhead=8, dropout=0.1)
        self.summary_attn = SummaryAttn(hidden_dim=config["lstm_hidden_size"], num_attn=3, config=["dropout"]) 
        self.context_fusion = nn.Linear(hidden_dim=config["lstm_hidden_size"], 

        self.v_proj = nn.Linear(
                config["img_feature_size"],
                config["lstm_hidden_size"]
            )

        self.j_proj = n.Linear(
                config["lstm_hidden_size"]*2,
                config["lstm_hidden_size"]
            )

        self.q_proj = n.Linear(
                config["lstm_hidden_size"]*2,
                config["lstm_hidden_size"]
            )

        self.h_proj = n.Linear(
                config["lstm_hidden_size"]*2,
                config["lstm_hidden_size"]
            )

    def lang_emb(self, seq, seq_len, lang_type='Ques'):
        rnn_cls = self.hist_rnn if lang_type =='Hist' else self.ques_rnn if lang_type == 'Ques' else None
        proj_cls = self.h_proj if lang_type =='Hist' else self.q_proj if lang_type == 'Ques' else None

        seq = self.word_embed(seq)
        seq_emb, _ = rnn_cls(seq, seq_len)

        seq_last_idx = seq_len.cpu().numpy()-1
        seq_fwd = seq_emb[range(seq.size(0)), seq_last_idx, :self.config["lstm_hidden_size"]]
        seq_bwd = seq_emb[:, 0, self.config["lstm_hidden_size"]:]

        seq_emb = proj_cls(torch.cat((seq_fwd, seq_bwd), dim=-1))
        seq_emb = seq_emb.unsqueeze(1)
        return seq_emb

    def add_entry(self, memory, hist, hist_len):
        h_emb = self.lang_emb(hist, hist_len, lang_type='Hist')
        if memory is None: memory = h_emb
        else: memory = torch.cat((memory, h_emb), 1)
        return memory

    def forward(self, batch):
        q = batch["ques"]      # b x 10 x 20
        ql = batch['ques_len'] # b x 10
        h = batch["hist"]      # b x 10 x 40
        hl = batch["hist_len"] # b x 10
        v = batch["img_feat"]  # b x 36 x 2048

        img_feat = self.v_proj(v)
        n_batch, n_round, _ = q.size()
        enc_outs = []
        structures = []
        hs = None

        for i in range(n_round):
            hs = self.add_entry(hs, h[:, i, :], hl[:, i])
        
        q_feat = self.lang_emb(q[:, i, :], ql[:, i]) # how to deal with question features?
        
        # q_feat: question features, hs: history features, img_feat: image features
        V = img_feat
        Q = q_feat
        R = hs

        # Run all feature tensors through two utility layers
        V, Q, R = self.util1(V, Q, R)
        V, Q, R = self.util2(V, Q, R)
        
        # Only keep visual and question features, run them through self-attention layer
        c_v = self.summary_attn(V, V)
        c_q = self.summary_attn(Q, Q)

        # Project visual context and question context vectors to single context vector
        out = torch.cat((c_v, c_q), dim=2)
        out = self.j_proj(out)

        return out

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
