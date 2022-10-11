import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from .encoder import Encoder
import math

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, h):
        super().__init__()
        self.d_model = d_model
        self.h = h

        assert d_model % self.h == 0
        self.d_k = d_model // self.h
        self.d_v = d_model // self.h

        self.q_fc = nn.Linear(d_model, d_model)  # (d_embed, d_model)
        self.k_fc = nn.Linear(d_model, d_model)  # (d_embed, d_model)
        self.v_fc = nn.Linear(d_model, d_model)  # (d_embed, d_model)
        self.out_fc = nn.Linear(d_model, d_model)  # (d_model, d_embed)

    def scaled_dot_product_attention(self, query, key, value):
        # query, key, value: (n_batch, seq_len, d_k)
        d_k = key.shape[-1]
        attention_score = torch.matmul(query, key.transpose(-2, -1))  # Q x K^T, (n_batch, seq_len, seq_len)
        attention_score = attention_score / math.sqrt(d_k)
        attention_prob = F.softmax(attention_score, dim=-1)  # (n_batch, seq_len, seq_len)
        out = torch.matmul(attention_prob, value)  # (n_batch, seq_len, d_k)
        return out

    def forward(self, query, key, value):
        # query, key, value: (n_batch, seq_len, d_model)
        # return value: (n_batch, h, seq_len, d_k)
        n_batch = query.size(0)

        def transform(x, fc):  # (n_batch, seq_len, d_embed)
            out = fc(x)  # (n_batch, seq_len, d_model)
            out = out.view(n_batch, -1, self.h, self.d_model//self.h)  # (n_batch, seq_len, h, d_k)
            out = out.transpose(1, 2)  # (n_batch, h, seq_len, d_k)
            return out

        query = transform(query, self.q_fc)  # (n_batch, h, seq_len, d_k)
        key = transform(key, self.k_fc)  # (n_batch, h, seq_len, d_k)
        value = transform(value, self.v_fc)  # (n_batch, h, seq_len, d_k)

        out = self.scaled_dot_product_attention(query, key, value)  # (n_batch, h, seq_len, d_k)
        out = out.transpose(1, 2)  # (n_batch, seq_len, h, d_k)
        out = out.contiguous().view(n_batch, -1, self.d_model)  # (n_batch, seq_len, d_model)
        out = self.out_fc(out)  # (n_batch, seq_len, d_model)
        return out


class Decoder(nn.Module):
    def __init__(self,
                 output_dim,
                 emb_dim,
                 hid_dim,
                 max_len,
                 n_layers,
                 dropout,
                 num_heads, #추가
                 sos_token=0,
                 padding_idx=None,):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.max_len = max_len
        self.n_layers = n_layers
        self.sos_token = sos_token

        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=padding_idx)
        self.rnn = nn.LSTM(emb_dim,
                           hid_dim,
                           n_layers,
                           dropout=dropout,
                           batch_first=True,)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention = MultiHeadAttention(hid_dim, num_heads)
        self.attn = nn.Linear(self.hid_dim * 2, self.max_len)
        self.attn_combine = nn.Linear(self.hid_dim * 2, self.hid_dim)


    def forward(self, hs, hidden, cell, trg=None, teacher_forcing_ratio=0):
        # target: [batch_size]
        # hidden: [n layers * n directions, batch_size, hid dim]
        # cell: [n layers * n directions, batch_size, hid dim]

        batch_size = hidden.shape[1]
        outputs = torch.zeros(batch_size, self.max_len, self.output_dim)
        # outputs = torch.zeros(batch_size, self.max_len, self.output_dim).to(self.device)

        # 첫 번째 입력값 <sos> 토큰
        if trg is None:
            assert teacher_forcing_ratio == 0, \
                'Without target data, teacher_forcing_ratio must be 0'

        decoder_input = torch.LongTensor([self.sos_token]).repeat(batch_size)

        for t in range(1, self.max_len):  # <eos> 제외하고 trg_len-1 만큼 반복

            decoder_input = decoder_input.view(-1, 1)  # input: [1, batch_size], 첫번째 input은 <SOS>

            embedded = self.embedding(decoder_input)
            embedded = self.dropout(embedded)  # [batch_size, 1, emd dim]

            output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
            # output: [batch_size, seq len, hid dim]
            # hidden: [n layers, batch size, hid dim]
            # cell: [n layers, batch size, hid dim]
            # hs: [src_len, batch_size, hid dim * n directions]

            #query, key, value: (n_batch, seq_len, d_model)
            value = hs.transpose(0, 1)
            key = hs.transpose(0, 1)
            query = output
            output = self.attention(query, key, value)

            #attention
            # batch_size = output.shape[0]
            # ht = hidden[0].view(batch_size, -1, 1)
            #
            # attn_weights = F.softmax(torch.bmm(hs, ht), dim=1)
            # attn_applied = torch.bmm(attn_weights.squeeze(2).unsqueeze(1), hs).squeeze(1)
            #
            # output = torch.cat((hidden[0], attn_applied), 1)
            # output = self.attn_combine(output)

            output = self.fc_out(output.squeeze(1))  # [batch size, output dim]

            outputs[:, t, :] = output
            top1 = output.argmax(1)

            teacher_force = random.random() < teacher_forcing_ratio
            decoder_input = trg[:, t] if teacher_force else top1

        return outputs

if __name__ == "__main__":
    # input_dim, emb_dim, hid_dim, n_layers, dropout, padding_idx = None
    encoder = Encoder(10, 512, 512, 1, 0.5, 1)
    source = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
    outputs, hidden, cell = encoder(source)
    decoder = Decoder(10, 512, 512, 4, 1, 0.5, 8)
    trg = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
    r = decoder(outputs, hidden, cell)
    print(r)
