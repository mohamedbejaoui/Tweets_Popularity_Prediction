import json
import numpy as np

from kafka import KafkaConsumer, KafkaProducer
from hawkes_point_process.marked_hawkes import create_start_points, fit_parameters, get_total_events

"""
	Kafka consumer that group received tweets into cascades, 
	fits the hawkes parameters for each received cascades 
	(if cascade size > 50 and T is reached)
	and then either send the cascades to a topic for training
	or predicts the cascades size using the latest trained RF model(inference)
	and send an alert to a topic if the predicted size of a cascade is > thresh
"""


CASCADE_OBSERVATION_TIME = 600 # 10min
ALERT_THRESH = 200 # we send an alert if a cascade size estimtation is bigger than this threshold
SEED = 0

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


cascades_info = {} # dict that stores params of each cascades to be fitted
fitted_cascades = set()

# create starting params for fitting hawkes process. same seed means same starting point for all cascades
start_params = np.array([np.random.choice(param) for param in create_start_points(seed=None).values()])

for tweet_info in consumer:
	# get cascade number from message key
	num_cascade = tweet_info.key.decode('utf-8').split('-')[0]
	# add received tweet's info to the current cascade history. First row of a cascade contains (None, None) that must be ignored before fitting the hawkes params
	cascades_info[num_cascade] = np.concatenate((cascades_info.get(num_cascade, np.array([None, None]).reshape((1, 2))), 
												 np.array([tweet_info.value['magnitude'], tweet_info.value['time']]).reshape((1, 2))), axis=0)

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

		# calculate the estimated final cascade size
		N_pred = get_total_events(history=history, T=CASCADE_OBSERVATION_TIME, params=fitted_params)['total']
		# TODO: add RF prediction layer

		# if the estimated size of the cascade is > 50, send an alert to a topic
		if N_pred >= 200:
			print(f"Sending alert for {num_cascade}")
			producer.send(topic='pred_size_alert',
						  value={'cascade_idx': num_cascade, 'estimated_size': N_pred})

		producer.flush()
