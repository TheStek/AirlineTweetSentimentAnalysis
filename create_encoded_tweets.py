import pandas as pd

def encode_cols(raw_data, colnames):

	data = raw_data.copy()

	# Loop through each of the columns inputted, get all the values for that column and assign each 
	# an integer in catmap. Then created a new column with the category values from catmap

	for cname in colnames:
		cats = data[cname].unique()
		cat_map = {cats[i] : i for i in range(len(cats))}
		print(f"{cname} mappings:\n{cat_map}")

		data[f"encoded_{cname}"] = data[cname].apply(lambda x: cat_map[x])

	return data


# Load the dataset, extract the hour created into a new column, encode the specified columns and save as encoded_tweets.csv

tweets = pd.read_csv("Tweets.csv")

tweets["hour"] = tweets["tweet_created"].apply(lambda x: int(x[11:13]))

enc_tweets = encode_cols(tweets, ["airline", "airline_sentiment", "negativereason"])

enc_tweets.to_csv("encoded_tweets.csv")