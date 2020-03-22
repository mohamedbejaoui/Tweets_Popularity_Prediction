import pandas as pd
import random
import json
import time
import numpy as np
from threading import Thread

from kafka import KafkaProducer


KAFKA_TOPIC = 'fit_hawkes_params'

# read tweets data
tweets_dir = './data/tweets'
data_df = pd.read_csv(tweets_dir+'/data.csv')
index_df = pd.read_csv(tweets_dir+'/index.csv')

tweets_sent_count = {cascade_idx: 0 for cascade_idx in index_df.index}

producer = KafkaProducer(bootstrap_servers='localhost:9092', 
						 client_id='pfe2019',
						 value_serializer=lambda x: json.dumps(x).encode('utf-8'))

# mean and variance for the delay distribution
DELAY_MEAN = 1
DELAY_VAR = 10


while tweets_sent_count != {}:
	cascade_idx = random.choice(list(tweets_sent_count.keys()))
	tweet_idx = tweets_sent_count[cascade_idx] + index_df.iloc[cascade_idx]['start_ind'] - 1
	tweets_sent_count[cascade_idx] += 1
	tweet_info = data_df.iloc[tweet_idx].to_dict()
	# if last tweet in chosen cascade is sent, remove cascade from data we sample from
	N_real = index_df.iloc[cascade_idx]['end_ind'] - index_df.iloc[cascade_idx]['start_ind'] + 1
	if tweets_sent_count[cascade_idx] == N_real:
		del tweets_sent_count[cascade_idx]

	# send sampled tweet's info
	producer.send(topic=KAFKA_TOPIC,
				  value=tweet_info,
				  # TODO: modify key when initial training dataset is constructed
				  key=f'cascade_n{cascade_idx+1}-{N_real}'.encode('utf-8'))

	# add random delay before sending next tweet
	time.sleep(abs(np.random.normal(DELAY_MEAN, DELAY_VAR, 1)[0]))


producer.flush()
