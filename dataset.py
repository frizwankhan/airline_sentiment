import pandas as pd
from torch.utils.data import Dataset, DataLoader
from preprocess import Preprocess
import torch

class CustomDataset(Dataset, Preprocess):

    def __init__(self, data, train=True, **kwargs):
        super().__init__(**kwargs)
        # data = pd.read_csv(path)
        texts = data["text"]
        labels = data["airline_sentiment"]
        self.texts, self.labels = self.return_preprocessed_data(texts, labels)

    def transform(self, text, label):
        return torch.tensor(text, dtype=torch.int), torch.tensor(label ,dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.transform(self.texts[idx], self.labels[idx])

if __name__ == "__main__":
    obj = CustomDataset(pd.read_csv("airline_sentiment_analysis.csv"), max_len = 25, demoji=False, do_stemming=False, do_lemming=True)
    print(len(obj))
    # loader = DataLoader(obj, batch_size=64, shuffle=True)
    # data, label = next(iter(loader))
    # print(data.size())
    # print(label.size())