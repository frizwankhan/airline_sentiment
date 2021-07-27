from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from prediction import Predict
import os

class Sentences(BaseModel):
    texts: list


app = FastAPI()


@app.get("/")
def home():
    return {"Rizwan":
                "This is my fast api that predicts airline sentiments based on text given to it"}


@app.get("/predict/{sentence}")
def predict_get(sentence: str):
    print(sentence)
    predict = Predict()
    y = predict.predict([sentence])
    return {sentence: y[0]}


@app.post("/predict")
def predict_post(sentences: Sentences):
    # print("hello")
    predict = Predict()
    y = predict.predict(sentences.texts)
    return {text: pred for text, pred in zip(sentences.texts, y)}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host='0.0.0.0', port=port)
