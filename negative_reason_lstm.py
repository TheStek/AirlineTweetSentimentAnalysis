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


tweets = pd.read_csv("encoded_tweets.csv", index_col = 0)

tweets = tweets.loc[tweets["airline_sentiment"] == "negative"]

train, test = train_test_split(tweets, test_size = 0.2)

tk = Tokenizer(num_words = 5000)
tk.fit_on_texts(train["text"])

train["text_seq"] = tk.texts_to_sequences(train["text"])
test["text_seq"] = tk.texts_to_sequences(test["text"])

max_length = max(map(len, train["text_seq"]))

text_train = np.asarray(pad_sequences(train["text_seq"], maxlen = max_length, padding = "post"))
text_test = np.asarray(pad_sequences(test["text_seq"], maxlen = max_length, padding = "post"))

vocab_size = len(tk.word_index) + 1

vocab_size

meta_train = np.asarray(list(zip(train["encoded_airline"], train["hour"])))
meta_test = np.asarray(list(zip(test["encoded_airline"], test["hour"])))

def get_model():
  text_in = Input(shape = (max_length, ), name = "Text")
  meta_in = Input(shape = (2, ), name = "Metadata")

  embedded = Embedding(vocab_size, 64, input_length = max_length, name = "Embedding")(text_in)


  conv = Conv1D(64, 3)(embedded)
  conv = Dropout(0.4)(conv)


  lstm = LSTM(32, dropout = 0.4)(conv)



  flat_text = Flatten()(lstm)

  combined = Concatenate(axis=-1)([flat_text, meta_in])

  dense_1 = Dense(16, activation = "relu")(combined)
  dense_1 = Dropout(0.4)(dense_1)

  dense_2 = Dense(32, activation = "relu")(dense_1)
  dense_2 = Dropout(0.2)(dense_2)


  out = Dense(11, activation = "softmax", name = "Sentiment")(dense_2)

  return Model([text_in, meta_in], out)


model = get_model()

model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["acc"])


model.fit([text_train, meta_train], np.asarray(train["encoded_negativereason"]), epochs=25)

model.evaluate([text_test, meta_test], np.asarray(test["encoded_negativereason"]))


model = Sequential()

model.add(Embedding(vocab_size, 64, input_length=34))
model.add(LSTM(32, dropout = 0.4))
model.add(Dense(16, activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(11, activation = "softmax"))

model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

model.fit(text_train, np.asarray(train["encoded_negativereason"]), epochs = 25)

model.evaluate(text_test, np.asarray(test["encoded_negativereason"]))

