import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from visdialch.utils import DynamicRNN
from .module import BackboneNetwork
from .net_utils import MLP
from torch.nn.utils.weight_norm import weight_norm

class GLAN(nn.Module):
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
        self.backbone = BackboneNetwork(dropout=config["model_dropout"])         
        self.v_proj = weight_norm(
            nn.Linear(
                config["img_feature_size"], 
                config["lstm_hidden_size"]
            ), dim=None
        )

        self.j_proj = weight_norm(
            nn.Linear(
                config["lstm_hidden_size"], 
                config["lstm_hidden_size"]
            ), dim=None
        )

        self.q_proj = weight_norm(
            nn.Linear(
                config["lstm_hidden_size"]*2, 
                config["lstm_hidden_size"]
            ), dim=None
        )

        self.h_proj = weight_norm(
            nn.Linear(
                config["lstm_hidden_size"]*2, 
                config["lstm_hidden_size"]
            ), dim=None
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
            q_feat = self.lang_emb(q[:, i, :], ql[:, i])
            z, disc = self.backbone(q_feat, hs, img_feat, i+1)
            # zero padding to make (n_batch x n_round x n_round) tensor
            disc = torch.mean(disc, 1) # mean value for all attention heads
            pad = Variable(torch.zeros(n_batch, 1, n_round-(i+1)).cuda())
            disc = torch.cat((disc, pad), dim=2)

            enc_outs.append(z)
            structures.append(disc)

        structure = torch.cat(structures, dim=1)
        enc_out = torch.cat(enc_outs, dim=1)
        enc_out = self.j_proj(enc_out)
        enc_out = torch.tanh(enc_out)
        return enc_out, structure
