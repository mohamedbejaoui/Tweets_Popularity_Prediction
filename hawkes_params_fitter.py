import json

from kafka import KafkaConsumer, KafkaProducer


"""
	Kafka consumer that group received tweets into cascades, 
	fits the hawkes parameters for each received cascades 
	(if cascade size > 50 and T is reached)
	and then either send the cascades to a topic for training
	or predicts the cascades size using the latest trained RF model(inference)
	and send an alert to a topic if the predicted size of a cascade is > thresh
"""


CASCADE_OBSERVATION_TIME = 300 # 5min

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

for tweet_info in consumer:
	num_cascade = tweet_info.key.decode('utf-8')
	cascades_info[num_cascade] = cascades_info.get(num_cascade, []) + [tweet_info.value]

	if len(cascades_info[num_cascade]) >= 50 and \
		cascades_info[num_cascade][-1]['time'] >= CASCADE_OBSERVATION_TIME and \
		num_cascade not in fitted_cascades:

		# fit hawkes params for this cascade
		# TODO: implement real optimizer for params fitting. For now it's just a dummy one
		fitted_hawkes_params = cascades_info[num_cascade][-1]['time'] * cascades_info[num_cascade][-1]['magnitude']

		fitted_cascades.add(num_cascade)

		print({'feature': fitted_hawkes_params, 'ts': cascades_info[num_cascade][-1]['time'], 'cascade_idx': num_cascade})

		# send the fitted params for a topic to train the RF model
		producer.send(topic='rf_train', 
					  value={'feature': fitted_hawkes_params, 'ts': cascades_info[num_cascade][-1]['time'], 'cascade_idx': num_cascade})
		producer.flush()