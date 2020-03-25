import json
import numpy as np
import requests
import time
import json
import pickle
import csv
from compress_pickle import dumps, loads
from threading import Thread, Lock

from kafka import KafkaConsumer, KafkaProducer
from hawkes_point_process.marked_hawkes import create_start_points, fit_parameters, get_total_events


"""
    Kafka node that receives tweets from fit_hawkes_params topic.
    If observation time for a cascade is reached, predict its final size.
    If it's bigger than a threshold, send and alert.
    ALso, this node checks whether old received cascades can be considered finished or not.
    If they are, we fit their hawkes params and save them into the rf training dataset
"""

CASCADE_OBSERVATION_TIME = 600 # 10min
ALERT_THRESH = 200 # we send an alert if a cascade size estimtation is bigger than this threshold
NUM_CASCADES_TO_CHECK = 10
CASCADE_FINISH_DURATION = 5 # if for a certain cascade, we don't receive a tweet after this duration, we consider it's finished (set to 1h)

tweet_consumer = KafkaConsumer('fit_hawkes_params', 
							  bootstrap_servers='localhost:9092', 
							  client_id='pfe2019',
							  auto_offset_reset='earliest',
							  enable_auto_commit=True,
							  auto_commit_interval_ms=1000, # commit every second
							  group_id='hawkes_params_fitters',
							  value_deserializer=lambda x: json.loads(x.decode('utf-8')))

# here, the goal is to create multiple consumer groups where each contain only one consumer.
rf_consumer = KafkaConsumer('pretrained_rf',
                            bootstrap_servers='localhost:9092',
                            client_id='pfe2019',
                            group_id='rf_consumer_group1')

producer = KafkaProducer(bootstrap_servers='localhost:9092',
						 client_id='pfe2019',
						 value_serializer=lambda x: json.dumps(x).encode('utf-8'))


# scaling factor: for new consumer groups that didn't receive the rf model yet from the kafka topic, we set an initial value of 1 for the scaling factor
scaling_factor = 1

# constants definition
CASCADE_OBSERVATION_TIME = 600 # 10min
ALERT_THRESH = 200 # we send an alert if a cascade size estimtation is bigger than this threshold
NUM_CASCADES_TO_CHECK = 10
CASCADE_FINISH_DURATION = 5 # if for a certain cascade, we don't receive a tweet after this duration, we consider it's finished (set to 1h)


# Thread1: update rf regressor
class rf_updater_thread(Thread):
	def run(self):
		global rf_regressor

		for rf_regressor in rf_consumer:
			print("Received new pre-trained random forest regressor")
			rf_regressor = loads(rf_regressor.value, compression="gzip")


# Thread2: process incoming tweets
class tweets_processor_thread(Thread):
	def run(self):		
		cascades_info = {} # dict that stores params of each cascades to be fitted
		cascades_last_ts = {} # stores for each cascade last time a tweet is received
		fitted_cascades = set() # stores fitted cascades numbers. Cascade which number is stored in this set won't be trated anymore
		cascades_for_train = [] # every time a tweet is received, we look into the oldest NUM_CASCADES_TO_CHECK cascades. for each, one we check if we can consider a cascade as finished or not
		cascade_considered_finished = []

		global scaling_factor

		# create starting params for fitting hawkes process. same seed means same starting point for all cascades
		start_params = np.array([np.random.choice(param) for param in create_start_points(seed=None).values()])

		for tweet_info in tweet_consumer:
		    # get cascade number from message key
		    num_cascade = tweet_info.key.decode('utf-8').split('-')[0]
		    # add received tweet's info to the current cascade history. First row of a cascade contains (None, None) that must be ignored before fitting the hawkes params
		    cascades_info[num_cascade] = np.concatenate((cascades_info.get(num_cascade, np.array([None, None]).reshape((1, 2))),
		              np.array([tweet_info.value['magnitude'], tweet_info.value['time']]).reshape((1, 2))), axis=0)
		    cascades_last_ts[num_cascade] = time.time()
		    if num_cascade not in cascades_for_train and num_cascade not in cascade_considered_finished:
		        cascades_for_train.append(num_cascade)


		    # As soon as we a received tweet's timestamp reaches the defined obsevation time, we fit the hawkes params and ignore upcoming tweets of same cascade 
		    if cascades_info[num_cascade][-1, 1] >= CASCADE_OBSERVATION_TIME and num_cascade not in fitted_cascades:
		        # fit hawkes params using current cascade's history
		        history = cascades_info[num_cascade][1:]
		        # if last tweet's timestamp is > observation_time, remove it
		        if history[-1, 1] > CASCADE_OBSERVATION_TIME:
		            history = history[:-1, :]
		        print(f"Fitting hawkes parameters for {num_cascade}")
		        result_params, _ = fit_parameters(history.astype(float), start_params)
		        # store fitted params in a dict
		        fitted_params = dict(K=result_params[0], beta=result_params[1], c=result_params[2], theta=result_params[3])
		        # add fitted params to set of cascade nums for which hawkes params were already fitted.
		        fitted_cascades.add(num_cascade)
		    
		        # compute the estimated final cascade size
		        prediction = get_total_events(history=history, T=CASCADE_OBSERVATION_TIME, params=fitted_params)
		        
		        if prediction['n_star'] >= 1: # super-critical regime
		            pass
		        else:
		            # RF prediction layer
		            rf_features = np.array([[result_params[2], result_params[3], prediction['a1'], prediction['n_star']]])
		            try:
		            	scaling_factor = rf_regressor.predict(rf_features)[0]
		            except NameError:
		            	pass
		            # improve cascade size estimation using predicted scaling factor
		            N_pred = len(history) + scaling_factor * prediction['a1'] / (1 - prediction['n_star'])
		            # if the estimated size of the cascade is > 50, send an alert to a topic
		            if N_pred >= ALERT_THRESH:
		                print(f"Sending alert for {num_cascade}")
		                producer.send(topic='pred_size_alert',
		                              value={'cascade_idx': num_cascade, 'estimated_size': N_pred})
		                producer.flush()


# create threads
thread1 = rf_updater_thread()
thread2 = tweets_processor_thread()

# run threads in the main thread
thread1.start()
thread2.start()
