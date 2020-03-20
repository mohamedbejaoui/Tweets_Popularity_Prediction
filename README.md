## Tweets popularity prediction

### How it works

In kafka_2.12-2.4.0 folder

* start zooKeeper server. In kafka directory run:<br>
```bin/zookeeper-server-start.sh config/zookeeper.properties```

* start kafka server. In kafka directory run:<br>
```bin/kafka-server-start.sh config/server.properties```

In our GitHub repository folder

* create necessary topics by running the following python script<br>
```python kafka_config.py```<br>
this script must be run only the first time this manipulaiton is done.<br>

* train the supervised model by running the following python script<br>
```python RF_model/random_forest_model.py```<br>

* host the trained model on Flask by running the following python script<br>
```python RF_model/flask_app.py```<br>

* run the kafka consumer that receives tweets to predict final cascade size<br>
```python hawkes_params_fitter.py```

* run the kafka consumer that receives alerts about cascade size estimations<br>
```python cascade_size_alert_receiver.py```

* finally, run the kafka producer that reads the tweets flow.<br>
```python tweets_data_reader.py```

[Quick command reference for Apache Kafka](https://gist.github.com/ursuad/e5b8542024a15e4db601f34906b30bb5)