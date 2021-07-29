import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from model import model_1
import torch
import torch.nn as nn
from dataset import CustomDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

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

    def eval_metrics_plotting(self, train_epoch_loss, train_epoch_acc,
                                    eval_epoch_loss, eval_epoch_acc,
                                    y_true, y_pred,
                                    y_true_train, y_pred_train):
        fig1 = plt.figure(1)
        plt.plot(list(range(len(train_epoch_loss))), train_epoch_loss, "b",  label="train_loss")
        plt.plot(list(range(len(eval_epoch_loss))), eval_epoch_loss, "g", label="eval_loss")
        plt.title("Train and Eval losses")
        plt.legend()

        fig2 = plt.figure(2)
        plt.plot(list(range(len(train_epoch_acc))), train_epoch_acc, "b", label="train_acc")
        plt.plot(list(range(len(eval_epoch_acc))), eval_epoch_acc, "g", label="eval_acc")
        plt.title("Train and Eval Accuracy")
        plt.legend()

        fig3 = plt.figure(3)
        fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred)
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label = "roc for evaluation")
        print("threshold ROC for train:")
        print(threshold)
        fpr, tpr, threshold = metrics.roc_curve(y_true_train, y_pred_train)
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label="roc for training")
        print("threshold ROC for train:")
        print(threshold)
        plt.legend()
        plt.title("roc curve")

        fig4 = plt.figure(4)
        precision, recall, threshold = metrics.precision_recall_curve(y_true, y_pred)
        print("threshold PR for eval:")
        print(threshold)
        plt.plot(precision, recall, label="PR curve for eval")
        precision, recall, threshold = metrics.precision_recall_curve(y_true_train, y_pred_train)
        plt.plot(precision, recall, label="PR curve for training")
        print("threshold PR for train:")
        print(threshold)
        plt.legend()
        plt.title("PR curve")

        print(f"fraction of positive examples i evalset is {sum(y_true)/len(y_true)}")
        print(f"fraction of positive examples i trainset is {sum(y_true_train)/len(y_true_train)}")
        print("Classification report for training set")
        print(metrics.classification_report(y_true_train, y_pred_train>0.5))
        print("Classification report for testing set")
        print(metrics.classification_report(y_true, y_pred>0.5))

        plt.show()



    def train_epoch(self, epoch):
        batch_losses = []
        batch_acc = []
        preds, truth = [], []
        with tqdm(self.train_loader, unit="batch") as tbatch:
            for batch, (x, y) in enumerate(tbatch):
                tbatch.set_description(f"Epoch {epoch+1}")
                truth = truth + y.detach().tolist()
                x = x.to(self.device)
                y = y.to(self.device)

                out = self.model(x).squeeze()

                preds = preds + out.detach().cpu().tolist()

                self.optimizer.zero_grad()
                batch_loss = self.loss(out, y.type(torch.float))
                batch_loss.backward()
                self.optimizer.step()

                acc = self.accuracy(out, y.type(torch.float))
                batch_losses.append(batch_loss.detach().cpu().numpy())
                batch_acc.append(acc)
                tbatch.set_postfix(batch_loss=batch_loss.detach().cpu().numpy(),
                                   batch_accuracy=acc)

        return batch_losses, batch_acc, preds, truth

    def eval_epoch(self):
        with torch.no_grad():
            batch_losses = []
            batch_acc = []
            truth, preds = [], []
            for batch, (x, y) in enumerate(self.test_loader):
                truth = truth + y.detach().tolist()

                x = x.to(self.device)
                y = y.to(self.device)

                out = self.model(x)[0]
                preds = preds + out.detach().cpu().tolist()

                batch_loss = self.loss(out, y.type(torch.float))

                acc = self.accuracy(out, y.type(torch.float))
                batch_losses.append(batch_loss.detach().cpu().numpy())
                batch_acc.append(acc)

            return batch_losses, batch_acc, preds, truth

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
            batch_losses, batch_acc, train_preds, train_truth = self.train_epoch(epoch)
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
            batch_losses, batch_acc, eval_preds, eval_truth = self.eval_epoch()
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

        return [train_epoch_loss, train_epoch_acc, eval_epoch_loss, eval_epoch_acc], [train_preds, train_truth, eval_preds, eval_truth]


if __name__ == "__main__":
    data = pd.read_csv("airline_sentiment_analysis.csv")
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    train_dataset = CustomDataset(train_data, max_len=25, demoji=False, do_stemming=False, do_lemming=True)
    test_dataset = CustomDataset(test_data, vocab_to_i=train_dataset.vocab_to_i)

    #pickling vocab_to_i object
    pkl_file = open("vocab_to_i.pkl", "wb")
    pickle.dump(train_dataset.vocab_to_i, pkl_file)

    obj = Train(train_dataset, test_dataset)
    loss_acc, pred_truth = obj.start_training(12)

    obj.eval_metrics_plotting(loss_acc[0], loss_acc[1],
                              loss_acc[2], loss_acc[3],
                              np.array(pred_truth[3]), np.array(pred_truth[2]),
                              np.array(pred_truth[1]), np.array(pred_truth[0]))
    print(len(pred_truth[0]), len(pred_truth[1]))

