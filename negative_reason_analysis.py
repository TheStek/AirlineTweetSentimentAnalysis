
import pandas as pd

import nltk
from nltk.tokenize import TweetTokenizer

import numpy as np

from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences

import sklearn
from nltk.classify.scikitlearn import SklearnClassifier
import re

nltk.download("stopwords")

tweets = pd.read_csv("encoded_tweets.csv", index_col = 0)
tweets = tweets.loc[tweets["airline_sentiment"] == "negative"]

def prepare_tweet(text, tk):
  return [word for word in tk.tokenize(text) if not re.match(r"@.*", word) and not word in nltk.corpus.stopwords.words("english")]

train, test = train_test_split(tweets, test_size = 0.2)

tk = TweetTokenizer()

train["text"] = train["text"].apply(lambda x: prepare_tweet(x, tk))
test["text"] = test["text"].apply(lambda x: prepare_tweet(x, tk))

train_vocab = nltk.FreqDist([w.lower() for t in train["text"] for w in t])

word_features = list(train_vocab)[:7500]

def find_features(document):
    words = set(document)
    features = []
    for w in word_features:
         features.append(int(w in words))

    return features

train["text"] = train["text"].apply(find_features)
test["text"] = test["text"].apply(find_features)

def generate_observation(row):
  x = row["text"]
  x.append(row["encoded_airline"])
  x.append(row["hour"])

  return x

x_train = list(train[["text", "encoded_airline", "hour"]].apply(generate_observation, axis = 1))
x_test = list(test[["text", "encoded_airline", "hour"]].apply(generate_observation, axis = 1))

y_train = list(train["encoded_negativereason"])
y_test = list(test["encoded_negativereason"])

from sklearn.naive_bayes import GaussianNB, BernoulliNB

m = BernoulliNB()
m.fit(x_train, y_train)
m.score(x_test, y_test)

import keras

nn = keras.models.Sequential([
  keras.layers.Dense(32, activation = "relu"),
  keras.layers.Dropout(0.2),
  keras.layers.Dense(16, activation = "relu"),
  keras.layers.Dropout(0.2),
  keras.layers.Dense(11, activation = "softmax")
])

nn.compile(optimizer = "adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

nn.fit(x_train, y_train, epochs = 15)

nn.evaluate(x_test, y_test)

from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

m = LinearSVC()
m.fit(x_train, y_train)
m.score(x_test, y_test)

