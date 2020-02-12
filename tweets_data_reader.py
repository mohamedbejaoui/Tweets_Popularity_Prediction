import pandas as pd
import random
import json

from kafka import KafkaProducer

train = True # TODO: change into variable that can be modified when calling this python script
KAFKA_TOPIC = 'fit_hawkes_params'

# read tweets data
tweets_dir = './data'
data_df = pd.read_csv(tweets_dir+'/data.csv')
data_df.index += 1
index_df = pd.read_csv(tweets_dir+'/index.csv')

tweets_sent_count = {cascade_idx: 0 for cascade_idx in index_df.index}

producer = KafkaProducer(bootstrap_servers='localhost:9092', 
						 client_id='pfe2019',
						 value_serializer=lambda x: json.dumps(x).encode('utf-8'))

# just for test, remove later
tweets_counter = 0
max_tweets = 1000

while tweets_sent_count != {} and tweets_counter < max_tweets:
	cascade_idx = random.choice(list(tweets_sent_count.keys()))
	tweet_idx = tweets_sent_count[cascade_idx] + index_df.iloc[cascade_idx]['start_ind']
	tweets_sent_count[cascade_idx] += 1
	tweet_info = data_df.iloc[tweet_idx].to_dict()
	# if last tweet in chosen cascade is sent, remove cascade from data we sample from and send end cascade info
	if tweets_sent_count[cascade_idx] == index_df.iloc[cascade_idx]['end_ind'] - index_df.iloc[cascade_idx]['start_ind']:
		del tweets_sent_count[cascade_idx]
		if train:
			tweet_info['isend_cascade'] = True

	# send sampled tweet's info
	producer.send(topic=KAFKA_TOPIC,
				  value=tweet_info,
				  key=f'cascade_n{cascade_idx+1}'.encode('utf-8'))

	tweets_counter += 1

producer.flush()
