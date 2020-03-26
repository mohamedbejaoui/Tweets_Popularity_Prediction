import numpy as np
import pandas as pd
import json
import csv
from compress_pickle import dumps, loads

from kafka import KafkaConsumer, KafkaProducer
from sklearn.ensemble import RandomForestRegressor


"""
Regularly, train the Random Forest Regressor.
Send the trained rf model to a kafka topic
"""

# KAFKA PRODUCER : send trained RF model to a topic to be loaded by the kafka inference node
producer = KafkaProducer(bootstrap_servers='localhost:9092',
            			client_id='pfe2019')

# KAFKA CONSUMER: receives new cascades to add to the training dataset
consumer = KafkaConsumer('rf_train',
                         bootstrap_servers='localhost:9092',
                         client_id='pfe2019',
                         auto_offset_reset='earliest',
                         enable_auto_commit=True,
                         auto_commit_interval_ms=1000, # commit every second
                         value_deserializer=lambda x: json.loads(x.decode('utf-8')))


NUM_SAMPLES_TRAIN = 500 # maximum number of cascades to train the RF on. If problem of topic max length is fixed, no need to set this!
NUM_NEW_CASCADES_FOR_TRAIN = 300 # number of new cascades received to trigger the random forest regressor training

new_cascades_counter = 0
new_cascades_info = pd.DataFrame(columns=['c','theta','A1','n_star','w'])

for new_cascade in consumer:
     new_cascades_counter += 1
     new_cascades_info.loc['new'+str(new_cascades_counter)] = new_cascade.value
     """
     Too much to open .csv for each single row

     # write cascade info at the end of rf training dataset csv file
     with open('./data/rf_train/training_data_rf.csv', 'a') as data_rf:
         writer = csv.writer(data_rf)
         writer.writerow([new_cascade_params['c'], new_cascade_params['theta'], new_cascade_params['a1'], new_cascade_params['n_star'], new_cascade_params['w_train']])
     """

     # train if we receive enough new training data points
     if new_cascades_counter >= NUM_NEW_CASCADES_FOR_TRAIN:
          # load training data
          data = pd.read_csv("./data/rf_train/training_data_rf.csv")
          # Merge with new data
          data = pd.concat([data,new_cascades_info], axis=0, sort=False)
          data.to_csv("./data/rf_train/training_data_rf.csv", index=False)
          new_cascades_info = pd.DataFrame(columns=['c','theta','A1','n_star','w']) # Reset

          # CLEAN
          data.drop(data[(data.w>5) | (data.w<0.005) | (data.n<5)].index, inplace=True)

          # choose NUM_SAMPLES_TRAIN random samples from the training dataset because the maximum length of a kafka message is limited
          train_idx = np.random.choice(data.index, min(NUM_SAMPLES_TRAIN, len(data)))
          x_train, y_train = data.loc[train_idx, ['c','theta','A1','n_star']].to_numpy(), data.loc[train_idx, 'w'].to_numpy()

          # define rf regressor
          rf_regressor = RandomForestRegressor(n_estimators = 100,
                                             criterion="mae", # "mse" Mean-Square Error; or "mae" Mean Absolute Error
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
          new_cascades_counter = 0
