import torch.nn as nn
import torch


class RNN(nn.Module):
    def __init__(self,
                 in_dim,
                 pad_idx,
                 embed_dim=128,
                 bidirectional=True,
                 hid_dim=256,
                 num_layers=2,
                 out_dim=4,
                 dropout=0.5):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(in_dim, embed_dim, pad_idx)
        self.rnn = nn.LSTM(embed_dim, hid_dim,
                           num_layers=num_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)
        self.fc1 = nn.Linear(hid_dim * num_layers, 64)
        self.fc2 = nn.Linear(64, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_len):
        embedded = self.dropout(self.embedding(text))
        text_len = text_len.to('cpu')
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_len)
        packed_out, (hidden, cell) = self.rnn(packed_embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        hidden = self.fc1(hidden)
        hidden = self.dropout(hidden)
        hidden = self.fc2(hidden)
        return hidden
