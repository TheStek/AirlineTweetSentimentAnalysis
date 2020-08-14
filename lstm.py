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

def encode_cols(data, colnames):
  for cname in colnames:
    cats = data[cname].unique()
    cat_map = {cats[i] : i for i in range(len(cats))}
    print(f"{cname} mappings:\n{cat_map}")

    data[f"encoded_{cname}"] = data[cname].apply(lambda x: cat_map[x])

tweets = pd.read_csv("encoded_tweets.csv")

train, test = train_test_split(tweets, test_size = 0.2)

tk = Tokenizer(num_words = 2500)
tk.fit_on_texts(train["text"])

vocab_size = len(tk.word_index) + 1


train["text_seq"] = tk.texts_to_sequences(train["text"])
test["text_seq"] = tk.texts_to_sequences(test["text"])

max_length = max(map(len, train["text_seq"]))

text_train = np.asarray(pad_sequences(train["text_seq"], maxlen = max_length, padding = "post"))
text_test = np.asarray(pad_sequences(test["text_seq"], maxlen = max_length, padding = "post"))


meta_train = np.asarray(list(zip(train["encoded_airline"], train["hour"])))
meta_test = np.asarray(list(zip(test["encoded_airline"], test["hour"])))

y_train = np.asarray(train["encoded_airline_sentiment"])
y_test = np.asarray(test["encoded_airline_sentiment"])

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






def get_meta_model():
  text_in = Input(shape = (max_length, ), name = "Text")
  meta_in = Input(shape = (2, ), name = "Metadata")

  embedded = Embedding(vocab_size, 32, input_length = max_length, name = "Embedding")(text_in)


  conv = Conv1D(32, 3)(embedded)
  conv = Dropout(0.4)(conv)


  lstm = LSTM(16, dropout = 0.4)(conv)



  flat_text = Flatten()(lstm)

  combined = Concatenate(axis=-1)([flat_text, meta_in])


  dense_2 = Dense(8, activation = "relu")(combined)
  dense_2 = Dropout(0.2)(dense_2)


  out = Dense(3, activation = "softmax", name = "Sentiment")(dense_2)

  return Model([text_in, meta_in], out)


meta_model = get_meta_model()

meta_model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["acc"])

meta_model.fit([text_train, meta_train], y_train, epochs=15)


print("Evaluating text and metadata LSTM: \n")

meta_model.evaluate([text_test, meta_test], y_test)

