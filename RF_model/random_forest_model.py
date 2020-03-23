#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pickle

# load training dataset into a pandas dataframe
data = pd.read_csv('data/rf_train/training_data_rf.csv')
data.drop(data[data.c>500].index, inplace=True)

x_train, y_train = data[['c','theta','A1','n_star']].to_numpy(), data['w'].to_numpy()

print(f"Training a Random Forest regressor on {len(x_train)} samples")

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

rf_regressor.fit(x_train, y_train)
      
## save model using pickle serializer
pickle.dump(rf_regressor, open("RF_model.pickle", 'wb'))
