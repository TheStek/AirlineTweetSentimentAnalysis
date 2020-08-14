import pandas as pd
import numpy as np




def encode_cols(raw_data, colnames):

	data = raw_data.copy()

	for cname in colnames:
		cats = data[cname].unique()
		cat_map = {cats[i] : i for i in range(len(cats))}
		print(f"{cname} mappings:\n{cat_map}")

		data[f"encoded_{cname}"] = data[cname].apply(lambda x: cat_map[x])

	return data


tweets = pd.read_csv("Tweets.csv")
tweets["hour"] = tweets["tweet_created"].apply(lambda x: int(x[11:13]))

enc_tweets = encode_cols(tweets, ["airline", "airline_sentiment", "negativereason"])

enc_tweets.to_csv("encoded_tweets.csv")