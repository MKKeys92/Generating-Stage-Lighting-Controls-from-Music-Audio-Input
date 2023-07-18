import numpy as np
import torch
import torch.nn as nn
from models.rnn_model import Encoder, get_subsequent_mask


class PureTransformerModel(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config


        encoder = Encoder(max_seq_len=self.config.max_seq_len,
                          input_size=self.config.prepro_config['music_dim'],
                          d_word_vec=self.config.music_emb_size,
                          n_layers=self.config.n_layers,
                          n_head=self.config.n_head,
                          d_k=self.config.d_k,
                          d_v=self.config.d_v,
                          d_model=self.config.d_model,
                          d_inner=self.config.d_inner,
                          dropout=self.config.dropout)

        self.encoder = encoder
        self.sliding_window_size = config.sliding_windown_size
        self.lambda_v = config.lambda_v
        self.device = device
        self.linear = nn.Linear(self.config.d_model,self.config.prepro_config['lighting_dim'])
        self.sigmoid = nn.Sigmoid()

    def forward(self, src_seq, src_pos, tgt_seq = None, epoch=None, is_training=False):
        bsz, seq_len, _ = src_seq.size()
        enc_mask = get_subsequent_mask(src_seq, self.sliding_window_size)
        enc_outputs, *_ = self.encoder(src_seq, src_pos, mask=enc_mask)
        outputs = self.linear(enc_outputs)
        outputs = self.sigmoid(outputs)

        return outputs


    def predict(self, src_seq, src_pos, tgt_seq = None):
        self.eval()
        with torch.no_grad():
            output = self(src_seq, src_pos)

        return output

    def train_step(self, src_seq, src_pos, tgt_seq, optimizer, criterion, epoch_i):

        output = self(src_seq, src_pos, tgt_seq=tgt_seq, epoch=epoch_i, is_training=True)

        # backward
        loss = criterion(output, tgt_seq)
        loss.backward()

        return loss