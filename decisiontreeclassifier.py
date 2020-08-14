# -*- coding: utf-8 -*-

import nltk
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


def tweet_to_words(tweettosplit):
    letters = re.sub("[^a-zA-Z]", " ", tweettosplit)
    words = letters.lower().split()
    stops = set(stopwords.words("english"))
    useful_words = [w for w in words if not w in stops]
    return (" ".join(useful_words))

df = pd.read_csv("Tweets.csv")

nltk.download('stopwords')
df['split_tweets']=df['text'].apply(lambda x: tweet_to_words(x))
train,test = train_test_split(df,test_size=0.2,random_state=42)
train_cleaned_tweets=[]
for tweet in train['split_tweets']:
    train_cleaned_tweets.append(tweet)
test_cleaned_tweets=[]
for tweet in test['split_tweets']:
    test_cleaned_tweets.append(tweet)
from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer(analyzer = "word")
train_features= v.fit_transform(train_cleaned_tweets)
test_features=v.transform(test_cleaned_tweets)
dense_features=train_features.toarray()
dense_test= test_features.toarray()
try:
    fit = DecisionTreeClassifier().fit(train_features,train['airline_sentiment'])
    pred = fit.predict(test_features)
except Exception:
    fit = DecisionTreeClassifier().fit(dense_features,train['airline_sentiment'])
    pred = fit.predict(dense_test)
accuracy = accuracy_score(pred,test['airline_sentiment'])
print('Accuracy of DecisionTreeClassifier is '+ str(accuracy))