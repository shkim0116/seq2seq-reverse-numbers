from model.encoder import Encoder
from model.decoder import Decoder
from model.seq2seq import Seq2Seq
from model.utils import *

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
src_pad_idx = 10
trg_pad_idx = 10
sos_token = 11
teacher_forcing_ratio = 0.5

test_dataloader = get_dataloader('data/test_data.txt', batch_size)

enc = Encoder(input_dim, enc_emb_dim, hid_dim, n_layers, enc_dropout, src_pad_idx)
dec = Decoder(output_dim, dec_emb_dim, hid_dim, max_len, n_layers, dec_dropout, sos_token, trg_pad_idx)
model = Seq2Seq(enc, dec, device).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

model.load_state_dict(torch.load('tut3-model.pt'))


def try_reversed_num(input_num):
    num_seq = [int(i) for i in input_num.split()]

    if len(num_seq) < 10:
        num_seq += [10] * (10 - len(num_seq))
    num_seq = [11] + num_seq + [12]
    num_seq = torch.LongTensor(num_seq)

    num_seq = num_seq.unsqueeze(0)
    num_seq = num_seq.to(device)

    output = model(num_seq)  # only src
    output_dim = output.shape[-1]
    output = output[:, 1:, :].view(-1, output_dim)
    result = ''
    for out in output:
        predict = out.argmax()
        if predict < 10:
            predict = predict.cpu()
            result += str(predict.numpy()) + " "
    if len(result) > 0:
        result = result[:-1]
    return result


if __name__ == "__main__":
    input_num = input('number sequence: ')
    result = try_reversed_num(input_num)
    print("reversed sequence: " + result)