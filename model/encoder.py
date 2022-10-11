import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, padding_idx=None, bidirectional=False):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=padding_idx)
        self.cnn1 = nn.Conv2d(1, 1, 2, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(2, stride=1)
        self.dropout1 = nn.Dropout(dropout)
        self.cnn2 = nn.Conv2d(1, 1, 2, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(2, stride=1)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: [batch_size, src_len]
        embedded = self.embedding(src)
        embedded = self.dropout(embedded)
        embedded = embedded.unsqueeze(dim=1)

        cnn1_output = self.cnn1(embedded)
        cnn1_output = self.maxpool1(cnn1_output)
        cnn1_output = self.dropout(cnn1_output)

        cnn2_output = self.cnn2(cnn1_output)
        cnn2_output = self.maxpool2(cnn2_output)
        cnn2_output = self.dropout(cnn2_output)

        cnn2_output = cnn2_output.squeeze(dim=1)
        outputs, (hidden, cell) = self.rnn(cnn2_output)

        # output: [src_len, batch_size, hid dim * n directions]
        # hidden: [n layers * n directions, batch_size, hid dim]
        # cell: [n layers * n directions, batch_size, hid dim]

        return outputs, hidden, cell

if __name__ == "__main__":
    # input_dim, emb_dim, hid_dim, n_layers, dropout, bidirectional=False, padding_idx=None
    encoder = Encoder(10, 32, 64, 1, 0.5, 1)
    source = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
    print(encoder(source))