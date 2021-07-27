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
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        self.linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding_mat(x)
        out, hidden = self.lstm(x)
        out = out[:, -1, :]
        out = self.sigmoid(self.linear(out))

        return out


if __name__ == "__main__":
    model = model_1(100, 23)
    x = np.random.randint(50, size=(10, 5))
    model(torch.tensor(x, dtype=torch.int))
