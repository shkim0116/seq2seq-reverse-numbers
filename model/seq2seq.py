import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            'Hidden dimensions of encoder decoder must be equal'
        assert encoder.n_layers == decoder.n_layers, \
            'Encoder and decoder must have equal number of layers'

    def forward(self, src, trg=None, teacher_forcing_ratio=0):

        outputs, hidden, cell = self.encoder(src)
        outputs = self.decoder(outputs, hidden, cell, trg, teacher_forcing_ratio)

        return outputs


# if __name__ == "__main__":
#     # 하이퍼 파라미터 지정
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#         print("GPU is available")
#     else:
#         device = torch.device("cpu")
#         print("GPU not available, CPU used")
#
#     input_dim = 10
#     output_dim = 10
#     enc_emb_dim = 32  # 임베딩 차원
#     dec_emb_dim = 32
#     hid_dim = 64  # hidden state 차원
#     n_layers = 1
#     enc_dropout = 0.5
#     dec_dropout = 0.5
#     max_len = 4
#
#     enc = Encoder(input_dim, enc_emb_dim, hid_dim, n_layers, enc_dropout)
#     dec = Decoder(output_dim, dec_emb_dim, hid_dim, max_len, n_layers, dec_dropout)
#     model = Seq2Seq(enc, dec, device).to(device)
#
#     source = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
#     print(model(source))
#     # # input_dim, emb_dim, hid_dim, n_layers, dropout, padding_idx = None
#     # encoder = Encoder(10, 16, 32, 1, 0.5, 1)
#     # source = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
#     # hidden, cell = encoder(source)
#     # decoder = Decoder(10, 16, 32, 4, 1, 0.5)
#     # trg = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
#     # decoder(hidden, cell)