import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

def pad(encoded, max_len):
    for line in encoded:
        if len(line) < max_len: # 현재 샘플이 정해준 길이보다 짧으면
            line += [10] * (max_len - len(line))
    for i, line in enumerate(encoded):
        encoded[i] = [11] + line + [12]
    return torch.LongTensor(encoded)


def get_dataloader(path, batch_size):
    total_src = []
    total_trg = []
    with open(path, 'r') as f:
        for line in f:
            splited = line.split('\t')
            src = splited[0].split()
            trg = splited[1].split()
            src = [int(i) for i in src]
            trg = [int(i) for i in trg]
            total_src.append(src)
            total_trg.append(trg)

    src_encoded = pad(total_src, 10)
    trg_encoded = pad(total_trg, 10)

    dataset = TensorDataset(src_encoded, trg_encoded)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    return dataloader


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


# 학습을 위한 함수
def train(model, iterator, optimizer, criterion, teacher_forcing_ratio):
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src, trg = batch
        src = src.to(device)
        trg = trg.to(device)

        optimizer.zero_grad()

        # output: [batch_size, seq len, hid dim * n directions]
        output = model(src, trg, teacher_forcing_ratio=teacher_forcing_ratio)
        output_dim = output.shape[-1]
        output = output[:, 1:, :].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


# evaluation function
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src, trg = batch
            src = src.to(device)
            trg = trg.to(device)

            # output: [trg len, batch size, output dim]
            output = model(src, trg, 0)  # teacher forcing off
            output_dim = output.shape[-1]
            output = output[:, 1:, :].reshape(-1, output_dim)  # [(trg len -1) * batch size, output dim]
            trg = trg[:, 1:].reshape(-1)  # [(trg len -1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


# function to count training time
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs