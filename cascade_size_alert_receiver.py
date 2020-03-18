import json

from kafka import KafkaConsumer


consumer = KafkaConsumer('pred_size_alert', 
						  bootstrap_servers='localhost:9092', 
						  client_id='pfe2019',
						  enable_auto_commit=True,
						  auto_commit_interval_ms=1000, # commit every second
						  group_id='cascade_size_alert_receiver',
						  value_deserializer=lambda x: json.loads(x.decode('utf-8')))

for alert in consumer:
	print(f"Received an alert for {alert.value['cascade_idx']}")
	print(f"Estimated cascade size: {alert.value['estimated_size']} \n")