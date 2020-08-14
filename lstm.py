import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split


from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Embedding, Dropout, Conv1D, Flatten, Input, LSTM, Concatenate
from keras.models import Sequential, Model


nltk.download("stopwords")


# Load in the encoded dataset (create_encoded_tweets.py must be run before this script) and split into train and test dataframes

tweets = pd.read_csv("encoded_tweets.csv")

train, test = train_test_split(tweets, test_size = 0.2)


# Create and fit the tokenizer on the train text using 2500 words

tk = Tokenizer(num_words = 2500)
tk.fit_on_texts(train["text"])

vocab_size = len(tk.word_index) + 1

# Convert the text into sequences of the encoded words

train["text_seq"] = tk.texts_to_sequences(train["text"])
test["text_seq"] = tk.texts_to_sequences(test["text"])

# Find the tweet with the longest length and pad all other tweets with 0s to be this length
# Save the padded tweets as the train and test text

max_length = max(map(len, train["text_seq"]))

text_train = np.asarray(pad_sequences(train["text_seq"], maxlen = max_length, padding = "post"))
text_test = np.asarray(pad_sequences(test["text_seq"], maxlen = max_length, padding = "post"))

# Extract the metadata from each dataframe and zip into tuples

meta_train = np.asarray(list(zip(train["encoded_airline"], train["hour"])))
meta_test = np.asarray(list(zip(test["encoded_airline"], test["hour"])))

# Extract train and test labels

y_train = np.asarray(train["encoded_airline_sentiment"])
y_test = np.asarray(test["encoded_airline_sentiment"])


# First model takes in only the text and is a sequential neural network

text_model = Sequential()

text_model.add(Embedding(vocab_size, 64, input_length=34))
text_model.add(LSTM(32, dropout = 0.4))
text_model.add(Dense(16, activation = "relu"))
text_model.add(Dropout(0.2))
text_model.add(Dense(3, activation = "softmax"))

text_model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])



text_model.fit(text_train, y_train, epochs = 15)


print("Evaluating text only LSTM : \n")

text_model.evaluate(text_test, y_test)


# This model takes in the text, passes it through a Conv1d and a LSTM then combines with
# the metadata into a dense layer 
# The keras functional api is used here to make a non-sequential NN

def get_meta_model():
  text_in = Input(shape = (max_length, ), name = "Text")
  meta_in = Input(shape = (2, ), name = "Metadata")

  embedded = Embedding(vocab_size, 32, input_length = max_length, name = "Embedding")(text_in)


  conv = Conv1D(32, 3)(embedded)
  conv = Dropout(0.4)(conv)


  lstm = LSTM(16, dropout = 0.4)(conv)

  flat_text = Flatten()(lstm)

  combined = Concatenate(axis=-1)([flat_text, meta_in])


  dense = Dense(8, activation = "relu")(combined)
  dense = Dropout(0.2)(dense)


  out = Dense(3, activation = "softmax", name = "Sentiment")(dense)

  return Model([text_in, meta_in], out)


meta_model = get_meta_model()

meta_model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["acc"])

meta_model.fit([text_train, meta_train], y_train, epochs=15)


print("Evaluating text and metadata LSTM: \n")

meta_model.evaluate([text_test, meta_test], y_test)

