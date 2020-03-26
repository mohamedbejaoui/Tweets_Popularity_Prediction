## Tweets popularity analysis in an Apache Kafka Architecutre

### How it works


In kafka_2.12-2.4.0 folder

* start zooKeeper server. In kafka directory run:<br>
```bin/zookeeper-server-start.sh config/zookeeper.properties```

* start kafka server. In kafka directory run:<br>
```bin/kafka-server-start.sh config/server.properties```

In our GitHub repository folder

* create necessary topics by running the following python script<br>
```python kafka_config.py```<br>
It is only necessary to run this script the first time this manipulaiton is done.<br>

* run the kafka node that trains the random forest regressor<br>
```python random_forest_trainer.py```<br>
NB: At first the scaling factor is equal to 1. When this node receives enough new data points and trains the random forest regressor for the first time,
the processing nodes will receive the pre-trained regressor to improve the cascades' size estimation. 

* run the kafka node that receives tweets to predict final cascade size<br>
```python hawkes_params_fitter.py```

* run the kafka consumer that receives alerts about cascade size estimations<br>
```python cascade_size_alert_receiver.py```

* finally, run the kafka producer that reads the tweets flow.<br>
```python tweets_data_reader.py```

[Quick command reference for Apache Kafka](https://gist.github.com/ursuad/e5b8542024a15e4db601f34906b30bb5)




### Libraries required


* Install libraries present in the requirement.txt file<br> 
```pip install -r requirements.txt```

* Install python layer for Kafka. For conda environment users:<br>
```conda install -c conda-forge kafka-python```<br>

* Install non-linear optimization Ipopt. For conda environment users:<br>
```conda install -c conda-forge ipopt```<br>



### Useful links


* [Quick command reference for Apache Kafka](https://gist.github.com/ursuad/e5b8542024a15e4db601f34906b30bb5)<br>
* [kafka-python API](https://kafka-python.readthedocs.io/en/master/apidoc/modules.html)
