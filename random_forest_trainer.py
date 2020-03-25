import numpy as np
import pandas as pd
from compress_pickle import dumps, loads

from kafka import KafkaProducer
from sklearn.ensemble import RandomForestRegressor


"""
Regularly, train the Random Forest Regressor.
Send the trained rf model to a kafka topic
"""

# KAFKA PRODUCER : send trained RF model to a topic to be loaded by the kafka inference node
producer = KafkaProducer(bootstrap_servers='localhost:9092',
            			client_id='pfe2019')

NUM_SAMPLES_TRAIN = 500

# load training data
data = pd.read_csv("./data/rf_train/training_data_rf.csv")
# choose NUM_SAMPLES_TRAIN random samples from the training dataset because the maximum length of a kafka message is limited
train_idx = np.random.choice(data.index, NUM_SAMPLES_TRAIN)
x_train, y_train = data.loc[train_idx, ['c','theta','A1','n_star']].to_numpy(), data.loc[train_idx, 'w'].to_numpy()

# define rf regressor
rf_regressor = RandomForestRegressor(n_estimators = 100,
                                   criterion="mae", # Mean-Square Error; or "mae" Mean Absolute Error
                                   max_depth = None,
                                   min_samples_split=2,
                                   min_samples_leaf=1,
                                   min_weight_fraction_leaf=0.,
                                   max_features="auto",
                                   max_leaf_nodes = None,
                                   min_impurity_decrease=0.,
                                   min_impurity_split=None,
                                   bootstrap=True,
                                   oob_score=False,
                                   n_jobs=None,
                                   random_state=None,
                                   verbose=0)

# fit rf regressor
rf_regressor.fit(x_train, y_train)

# send trained model to the inference topic
print(f"Finished training Random Forest Regressor on {x_train.shape[0]} samples")
producer.send(topic='pretrained_rf',
              value=dumps(rf_regressor, compression="gzip"))
producer.flush()
