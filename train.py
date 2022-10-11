import time
import math
import torch
import torch.nn as nn
from torch import optim
from model.encoder import Encoder
from model.decoder import Decoder
from model.seq2seq import Seq2Seq
from model.utils import *

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

batch_size = 5
input_dim = 13
output_dim = 13
enc_emb_dim = 32  # 임베딩 차원
dec_emb_dim = 32
hid_dim = 64  # hidden state 차원
n_layers = 1
enc_dropout = 0.5
dec_dropout = 0.5
max_len = 12
num_heads = 8
src_pad_idx = 10
trg_pad_idx = 10
sos_token = 11
teacher_forcing_ratio = 0.5

train_dataloader = get_dataloader('data/train_data.txt', batch_size)
valid_dataloader = get_dataloader('data/valid_data.txt', batch_size)

enc = Encoder(input_dim, enc_emb_dim, hid_dim, n_layers, enc_dropout, src_pad_idx)
dec = Decoder(output_dim, dec_emb_dim, hid_dim, max_len, n_layers, dec_dropout, num_heads, sos_token, trg_pad_idx)
model = Seq2Seq(enc, dec, device).to(device)
model.apply(init_weights)
model.load_state_dict(torch.load('tut_cnn2-model.pt'))

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)


# 학습 시작
num_epochs = 50

best_valid_loss = float('inf')

for epoch in range(num_epochs):

    start_time = time.time()

    train_loss = train(model, train_dataloader, optimizer, criterion, teacher_forcing_ratio)
    valid_loss = evaluate(model, valid_dataloader, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'model_data/tut_cnn2-model.pt')

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')