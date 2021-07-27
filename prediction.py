import torch
from model import model_1
import pickle
import numpy as np
from preprocess import Preprocess


class Predict():

    def __init__(self):
        file = open("vocab_to_i.pkl", "rb")
        self.vocab_to_i = pickle.load(file)
        self.model = model_1(vocab_size=len(self.vocab_to_i) + 1,
                             max_len=25)
        self.model.load_state_dict(torch.load("model_1.pt", map_location=torch.device("cpu")))
        self.model.to(torch.device("cpu"))
        self.model.eval()
        self.process = Preprocess(vocab_to_i=self.vocab_to_i)

    def predict(self, x):
        x, _ = self.process.return_preprocessed_data(x)
        x = torch.tensor(x, dtype=torch.int)
        # y = self.model(x).detach().cpu()
        y = (self.model(x).detach().cpu() > 0.5).numpy()
        y = ["positive" if pred else "negative" for pred in y]
        return y


if __name__ == "__main__":
    x = ["flight was awesome", "worst"]
    predict = Predict()
    y = predict.predict(x)
    print(y)
