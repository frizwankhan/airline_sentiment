import pandas as pd

from model import model_1
import torch
import torch.nn as nn
from dataset import CustomDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from tqdm import tqdm
import pickle

class Train():

    def __init__(self, train_dataset, test_dataset):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model_1(vocab_size=len(train_dataset.vocab_to_i)+1,
                             max_len=train_dataset.max_len,
                             embedding_dim=64,
                             hidden_size=128,
                             num_layers=3).to(self.device)
        self.loss = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    def accuracy(self, out, y):
        total_correct = ((out > 0.5) == y).sum().detach().cpu().numpy()
        return total_correct / y.size(0)

    def train_epoch(self, epoch):
        batch_losses = []
        batch_acc = []
        with tqdm(self.train_loader, unit="batch") as tbatch:
            for batch, (x, y) in enumerate(tbatch):
                tbatch.set_description(f"Epoch {epoch+1}")
                x = x.to(self.device)
                y = y.to(self.device)

                out = self.model(x).squeeze()

                self.optimizer.zero_grad()
                batch_loss = self.loss(out, y.type(torch.float))
                batch_loss.backward()
                self.optimizer.step()

                acc = self.accuracy(out, y.type(torch.float))
                batch_losses.append(batch_loss.detach().cpu().numpy())
                batch_acc.append(acc)
                tbatch.set_postfix(batch_loss=batch_loss.detach().cpu().numpy(),
                                   batch_accuracy=acc)

        return batch_losses, batch_acc

    def eval_epoch(self):
        with torch.no_grad():
            batch_losses = []
            batch_acc = []
            for batch, (x, y) in enumerate(self.test_loader):
                x = x.to(self.device)
                y = y.to(self.device)

                out = self.model(x)[0]
                batch_loss = self.loss(out, y.type(torch.float))

                acc = self.accuracy(out, y.type(torch.float))
                batch_losses.append(batch_loss.detach().cpu().numpy())
                batch_acc.append(acc)

            return batch_losses, batch_acc

    def start_training(self, epochs):
        train_epoch_loss = []
        train_epoch_acc = []
        train_batch_all_losses = []
        train_batch_all_acc = []
        eval_epoch_loss = []
        eval_epoch_acc = []
        eval_batch_all_losses = []
        eval_batch_all_acc = []
        self.model.to(self.device)

        for epoch in range(epochs):
            ##### TRAINING #####
            self.model.train()
            batch_losses, batch_acc = self.train_epoch(epoch)
            print(f"epoch {epoch + 1} train loss is {sum(batch_losses) / len(batch_losses)}")
            print(f"epoch {epoch + 1} train acc is {sum(batch_acc) / len(batch_acc)}")
            print("all trainning losses, acc", end=" : ")
            print(train_epoch_loss, train_epoch_acc)

            train_epoch_loss.append(sum(batch_losses) / len(batch_losses))
            train_epoch_acc.append(sum(batch_acc) / len(batch_acc))

            train_batch_all_losses = train_batch_all_losses + batch_losses
            train_batch_all_acc = train_batch_all_acc + batch_acc
            ##### TRAINING #####

            ##### EVALUATION #####
            self.model.eval()
            batch_losses, batch_acc = self.eval_epoch()
            print(f"epoch {epoch + 1} eval loss is {sum(batch_losses) / len(batch_losses)}")
            print(f"epoch {epoch + 1} eval acc is {sum(batch_acc) / len(batch_acc)}")
            print("all eval losses, acc", end=" : ")
            print(eval_epoch_loss, eval_epoch_acc)
            eval_epoch_loss.append(sum(batch_losses) / len(batch_losses))
            eval_epoch_acc.append(sum(batch_acc) / len(batch_acc))

            eval_batch_all_losses = eval_batch_all_losses + batch_losses
            eval_batch_all_acc = eval_batch_all_acc + batch_acc
            ##### EVALUATION #####

        print("TRAINING OVER")

        self.model.eval()
        torch.save(self.model.state_dict(), "model_1.pt")


if __name__ == "__main__":
    data = pd.read_csv("airline_sentiment_analysis.csv")
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    train_dataset = CustomDataset(train_data, max_len=25, demoji=False, do_stemming=False, do_lemming=True)
    test_dataset = CustomDataset(test_data, vocab_to_i=train_dataset.vocab_to_i)

    #pickling vocab_to_i object
    pkl_file = open("vocab_to_i.pkl", "wb")
    pickle.dump(train_dataset.vocab_to_i, pkl_file)

    obj = Train(train_dataset, test_dataset)
    obj.start_training(20)
