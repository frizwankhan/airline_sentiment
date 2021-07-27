import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import string
from collections import Counter

from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


class Preprocess():
    def __init__(self, max_len=25, demoji=False, do_stemming=False, do_lemming=True, vocab_to_i=None):
        self.demoji = demoji
        self.do_stemming = do_stemming
        self.do_lemming = do_lemming
        self.max_len = max_len
        self.vocab_to_i = vocab_to_i

    # changing the texrs to lowercase
    def make_lower(self, text):
        return text.lower()

    # removing tags
    def remove_tags(self, text):
        text = re.sub("(\s#\w+ )|\s#\w+|#\w*\s", " ", text)
        return re.sub(r"@\S+\s", "", text)

    # removing emojis
    def demojify(self, text):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   u"\U00010000-\U0010ffff"
                                   "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)
        #     text = re.sub("[^a-z0-9\s]","", text)
        return (emoji_pattern.sub(r'', text))

    # remove http, www links
    def remove_http(self, text):
        return re.sub(r"http\S+|https\S+|\www\S+", "", text)

    def remove_punc(self, text):
        return text.translate(str.maketrans("", "", string.punctuation))

    def remove_stop(self, text):
        tokens = nltk.tokenize.word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words]
        return " ".join(tokens)

    def stemming(self, text):
        tokens = nltk.tokenize.word_tokenize(text)
        stemmer = PorterStemmer()
        stemmed_token = [stemmer.stem(word) for word in tokens]
        return " ".join(stemmed_token)

    def lemming(self, text):
        tokens = nltk.tokenize.word_tokenize(text)
        lemmer = WordNetLemmatizer()
        lemmed_token = [lemmer.lemmatize(word) for word in tokens]
        return " ".join(lemmed_token)

    def tokenize(self, sentence):
        words = nltk.tokenize.word_tokenize(sentence)
        token_sentence = []
        for word in words:
            if word not in self.vocab_to_i:
                token_sentence.append(0)
            else:
                token_sentence.append(self.vocab_to_i[word])
        # token = [self.vocab_to_i[word] for word in token]
        return token_sentence

    def return_tokenized_text(self, texts):
        if self.vocab_to_i is None:
            count = Counter(nltk.tokenize.word_tokenize(" ".join(list(texts))))
            self.vocab_to_i = {w: i + 1 for i, (w, c) in enumerate(count.most_common())}
        tokenized_text = []
        for sentence in texts:
            tokenized_text.append(self.tokenize(sentence))
        return tokenized_text

    def padding_token(self, tokens, max_len):
        new_tokens = np.zeros((len(tokens), max_len))
        for i, sentence in enumerate(tokens):
            if len(sentence) < max_len:
                sentence = sentence + list(np.zeros(max_len - len(sentence)))
                new_tokens[i, :] = sentence
            elif len(sentence) >= max_len:
                sentence = sentence[:max_len]
                new_tokens[i, :] = sentence
        return new_tokens

    def process_token(self, label):
        if label == "positive":
            return 1
        else:
            return 0

    def return_preprocessed_data(self, texts, labels=None):
        """
        input: A pandas series "texts" or numpy array or python list which is a series of all the tweets,
                labels

        output: preprocessed text in form of list of size (n_texts x max_len)

        """
        texts = pd.Series(texts)
        texts = texts.apply(self.make_lower)  # making_lowercase
        texts = texts.apply(self.remove_tags)  # removing tags like #, @
        texts = texts.apply(self.remove_http)  # removing http links
        if self.demoji:  # removing emoji
            texts = texts.apply(self.demojify)
        texts = texts.apply(self.remove_punc)  # removing punctuation
        texts = texts.apply(self.remove_stop)  # removing stop words
        if self.do_stemming:  # Stemming
            texts = texts.apply(self.stemming)

        if self.do_lemming:  # lemming
            texts = texts.apply(self.lemming)

        texts = self.return_tokenized_text(texts)  # tokenizing the texts
        texts = self.padding_token(texts, self.max_len)

        if labels is not None:
            labels = pd.Series(labels)
            labels = labels.apply(self.process_token).values

        return texts, labels


if __name__ == "__main__":
    obj = Preprocess()
    data = pd.read_csv("airline_sentiment_analysis.csv")
    texts = data["text"].values
    labels = data["airline_sentiment"]
    print(obj.return_preprocessed_data(texts, labels))
    print(obj.vocab_to_i)
