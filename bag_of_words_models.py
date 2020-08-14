import pandas as pd

import nltk
from nltk.tokenize import TweetTokenizer

import numpy as np

from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences

import sklearn
import re

nltk.download("stopwords")

tweets = pd.read_csv("encoded_tweets.csv")

# Tokenize the inputted tweet into words using the TweetTokenizer, removing any words starting with '@' or any in the 
# english stopwords set

def prepare_tweet(text, tk):
  return [word for word in tk.tokenize(text) if not re.match(r"@.*", word) and not word in nltk.corpus.stopwords.words("english")]

train, test = train_test_split(tweets, test_size = 0.2)

tk = TweetTokenizer()

train["text"] = train["text"].apply(lambda x: prepare_tweet(x, tk))
test["text"] = test["text"].apply(lambda x: prepare_tweet(x, tk))

# Find the 7500 most frequent words in the train text and record these in the list word_features

train_vocab = nltk.FreqDist([w.lower() for t in train["text"] for w in t])

word_features = list(train_vocab)[:7500]

# Take in a piece of text and record if it contains each word in the word_features list (in order)

def find_features(document):
    words = set(document)
    features = []
    for w in word_features:
         features.append(int(w in words))

    return features

train["text"] = train["text"].apply(find_features)
test["text"] = test["text"].apply(find_features)

# Take in a row of a dataframe and append the airline and hour to the text list to create a single observation vector

def generate_observation(row):
  x = row["text"]
  x.append(row["encoded_airline"])
  x.append(row["hour"])

  return x

# Apply the generate_observation function on the train and test dataframes to create the x_train and x_test data

x_train = list(train[["text", "encoded_airline", "hour"]].apply(generate_observation, axis = 1))
x_test = list(test[["text", "encoded_airline", "hour"]].apply(generate_observation, axis = 1))

# Extract the train and test labels

y_train = list(train["encoded_airline_sentiment"])
y_test = list(test["encoded_airline_sentiment"])


# Fit various models from sklearn on the generated data and evaluate them


from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

print("Training and evaluating on SGD Classifier and Logistic Regression")


for model in (SGDClassifier, LogisticRegression):
  m = model()
  m.fit(x_train, y_train)
  print(m.score(x_test, y_test))

print("Training and Evaluating using Linear SVC")

m = LinearSVC()
m.fit(x_train, y_train)
m.score(x_test, y_test)

import sklearn.naive_bayes as nb

# Fit each type of naive bayes model and assign a name to each 

gnb = nb.GaussianNB()
gnb.name_ = "Gaussian"

mnb = nb.MultinomialNB()
mnb.name_ = "Multinomial"

cnb = nb.ComplementNB()
cnb.name_ = "Complement"

bnb = nb.BernoulliNB()
bnb.name_ = "Bernoulli"

catnb = nb.CategoricalNB()
catnb.name_ = "Categorical"

# Loop through each nb model and find its test accuracy 
for m in (gnb, mnb, cnb, bnb, catnb):

  print(f"Training {m.name_} NB")

  m.fit(x_train, y_train)

  try:
    print(f"{m.name_} NB accuracy : {m.score(x_test, y_test)}")
  except IndexError:
    print(f"Couldn't test {m.name_} NB")


# Pass the bag of words data into a simple dense neural network

import keras

model = keras.models.Sequential([
  keras.layers.Dense(128, activation="relu"),
  keras.layers.Dropout(0.2),
  keras.layers.Dense(64, activation="relu"),
  keras.layers.Dropout(0.4),
  keras.layers.Dense(3, activation="softmax")
])

model.compile(optimizer = "adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(x_train, y_train, epochs = 15)

model.evaluate(x_test, y_test)

