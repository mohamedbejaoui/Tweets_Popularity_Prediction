## Tweets popularity prediction

### how it works

* start kafka server. In kafka directory run:<br>
```bin/kafka-server-start.sh config/server.properties```

* create necessary topics by running the following python script<br>
```python kafka_config.py```<br>
this script must be run only the first time this manipulaiton is done.<br>

* run the kafka consumer that receives tweets to predict final cascade size<br>
```python hawkes_params_fitter.py```

* run the kafka consumer that receives alerts about cascade size estimations<br>
```python cascade_size_alert_receiver.py```

* finally, run the kafka producer that reads the tweets flow.<br>
```python tweets_data_reader.py```

