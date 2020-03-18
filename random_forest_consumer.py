#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np

from kafka import KafkaConsumer, KafkaProducer
from sklearn.ensemble import RandomForestClassifier


"""
Get fitted parameters {c, theta, A1, n_star} of a cascade from appropriate topic.
Send to the Random Forest Classifier which predicts scaling parameter w
to compute N_estimated.
"""


## KAFKA CONSUMER : get the fitted parameters from topic 'hawkes_params_fitters'
consumer = KafkaConsumer('', 
                             bootstrap_servers='localhost:9092', 
						  client_id='pfe2019',
						  auto_offset_reset='earliest',
						  enable_auto_commit=True,
						  auto_commit_interval_ms=1000, # commit every second
						  group_id='hawkes_params_fitters',
						  value_deserializer=lambda x: json.loads(x.decode('utf-8')))

## KAFKA PRODUCER : send total retweets estimated in real time of a given cascade to a consumer
producer = KafkaProducer(bootstrap_servers='localhost:9092',
						 client_id='pfe2019',
						 value_serializer=lambda x: json.dumps(x).encode('utf-8'))


## SUPERVISED LEARNING 
model = RandomForestClassifier(n_estimators = 200,
                               criterion = 'gini',
                               max_depth = None,
                               max_leaf_nodes = None)

data = pd.read("./hawkes_point_process/training_data_RF.csv")
X, y = data[['c','theta','A1','n_star']], data['w']
model.fit(X, y)

# Real time fitting of consumed data
for fitted_params in consumer:
	# get cascade number from message key
	num_cascade = fitted_params.key.decode('utf-8').split('-')[0]
	
    """
    """
    c, theta, A1, n_star = 0.,0.,0.,0.
    X_test = np.array([c, theta, A1, n_star])
    w_pred = model.predict(X_test)
    N_pred = 
    
    producer.send(topic='pred_size_alert',
                  value={'cascade_idx': num_cascade, 'estimated_size': N_pred})
    producer.flush()


###