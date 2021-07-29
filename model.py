import torch
import torch.nn as nn
import numpy as np


class model_1(nn.Module):

    def __init__(self, vocab_size, max_len, embedding_dim=64, hidden_size=128, num_layers=3):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        self.embedding_mat = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional = True)

        self.linear_1 = nn.Linear(hidden_size*2, 64)
        self.linear_2 = nn.Linear(64,1)
        self.drop = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding_mat(x)
        out, hidden = self.lstm(x)
        out = out[:, -1, :]
        out = self.relu(self.linear_1(out))
        out = self.drop(out)
        out = self.sigmoid(self.linear_2(out))

        return out


if __name__ == "__main__":
    model = model_1(100, 25)
    x = np.random.randint(50, size=(10, 5))
    model(torch.tensor(x, dtype=torch.int))
