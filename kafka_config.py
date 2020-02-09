from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError


admin_client = KafkaAdminClient(bootstrap_servers="localhost:9092", client_id='pfe2019')

# for now, we only have one broker
# when possibility to have more brokers is avaialble, change replciation factor to 2
topics_list = [
	NewTopic(name='fit_hawkes_params', num_partitions=100, replication_factor=1)
]

# if topic doesn't exist create it, else delete it and recreate it 
try:
	admin_client.create_topics(new_topics=topics_list)
except TopicAlreadyExistsError:
	admin_client.delete_topics([topic.name for topic in topics_list])
	admin_client.create_topics(new_topics=topics_list)
