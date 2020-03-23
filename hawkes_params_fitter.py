import json
import numpy as np
import requests
import time
import pickle
import csv

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
CASCADE_FINISH_DURATION = 3600 # if for a certain cascade, we don't receive a tweet after this duration, we consider it's finished (set to 1h)

consumer = KafkaConsumer('fit_hawkes_params', 
						  bootstrap_servers='localhost:9092', 
						  client_id='pfe2019',
						  auto_offset_reset='earliest',
						  enable_auto_commit=True,
						  auto_commit_interval_ms=1000, # commit every second
						  group_id='hawkes_params_fitters',
						  value_deserializer=lambda x: json.loads(x.decode('utf-8')))

producer = KafkaProducer(bootstrap_servers='localhost:9092',
						 client_id='pfe2019',
						 value_serializer=lambda x: json.dumps(x).encode('utf-8'))

## FLASK API where Random Forest model is stored
URL = 'http://0.0.0.0:5000/api/'
HEADERS = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}



cascades_info = {} # dict that stores params of each cascades to be fitted
cascades_last_ts = {} # stores for each cascade last time a tweet is received
fitted_cascades = set() # stores fitted cascades numbers. Cascade which number is stored in this set won't be trated anymore
cascades_for_train = [] # every time a tweet is received, we look into the oldest NUM_CASCADES_TO_CHECK cascades. for each, one we check if we can consider a cascade as finished or not

# create starting params for fitting hawkes process. same seed means same starting point for all cascades
start_params = np.array([np.random.choice(param) for param in create_start_points(seed=None).values()])

for tweet_info in consumer:
    # get cascade number from message key
    num_cascade = tweet_info.key.decode('utf-8').split('-')[0]
    # add received tweet's info to the current cascade history. First row of a cascade contains (None, None) that must be ignored before fitting the hawkes params
    cascades_info[num_cascade] = np.concatenate((cascades_info.get(num_cascade, np.array([None, None]).reshape((1, 2))),
              np.array([tweet_info.value['magnitude'], tweet_info.value['time']]).reshape((1, 2))), axis=0)
    cascades_last_ts[num_cascade] = time.time()
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
        
        if prediction['n_star']>=1: # Cas du régime super-critique
            pass
        else:
            # RF prediction layer
            rf_features = json.dumps([[result_params[2], result_params[3], prediction['a1'], prediction['n_star']]])
            r = requests.post(URL, data=rf_features, headers=HEADERS)
            r_json = r.json()
            
            # Output : scaling factor w
            w_pred = float(json.loads(r_json[1:-1])[0])
            # improve cascade size estimation using predicted scaling factor
            N_pred = len(history) + w_pred * prediction['a1'] / (1 - prediction['n_star'])
            
            # if the estimated size of the cascade is > 50, send an alert to a topic
            if N_pred >= ALERT_THRESH:
                print(f"Sending alert for {num_cascade}")
                producer.send(topic='pred_size_alert',
                              value={'cascade_idx': num_cascade, 'estimated_size': N_pred})
                producer.flush()

    # check if we can consider some cascades as finished. If yes, we fit the hawkes params and save them in the RF training dataset
    cascade_idx_to_remove = []
    for i in range(NUM_CASCADES_TO_CHECK):
        if time.time() - cascades_last_ts[cascades_for_train[i]] >= CASCADE_FINISH_DURATION:
            # include only cascades whith more than 50 tweets in the training dataset
            cascade_len = len(cascades_info[cascades_for_train[i]])
            if  cascade_len >= 50:
                # fit hawkes params using current cascade's history
                history = cascades_info[num_cascade][1:]
                history = history[history[: 1] <= CASCADE_OBSERVATION_TIME]
                result_params, _ = fit_parameters(history.astype(float), start_params)
                fitted_params = dict(K=result_params[0], beta=result_params[1], c=result_params[2], theta=result_params[3])
                other_prams = get_total_events(history=history, T=CASCADE_OBSERVATION_TIME, params=fitted_params)
                rf_features = [result_params[2], result_params[3], other_prams['a1'], other_prams['n_star']]
                w_train = (cascade_len - len(history)) * (1 - n_star) / other_prams['a1']
                # write cascade info at the end of rf training dataset csv file
                with open('./data/rf_train/training_data_rf.csv', 'a') as data_rf:
                    writer = csv.writer(data_rf)
                    writer.writerow(rf_features + [w_train, cascade_len])
            cascade_idx_to_remove.append(i)
    # remove cascades that are saved for training from the list to check from
    for i in cascade_idx_to_remove:
        cascades_for_train.pop(i)
