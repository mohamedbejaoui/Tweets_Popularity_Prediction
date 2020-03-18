#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from tqdm import tqdm
from marked_hawkes import create_start_points, fit_parameters, get_total_events


## COMPLETE DATA HISTORY
indexes = pd.read_csv('index.csv')
cascades = pd.read_csv('data.csv', names=['time', 'magnitude'], header=0)
cascades = cascades[['magnitude','time']]
cascades.index += 1 # Match with the index.csv values

indexes = indexes.iloc[:10]

## CONTEXT PARAMETERS
observation_duration = 900 # seconds
start_params = np.array([np.random.choice(param) for param in create_start_points(0).values()]) #np.array([.8, .6, 100, .8])


## TRAINING DATA FOR RANDOM FOREST
RF_train_data = pd.DataFrame(columns=['c','theta','A1','n_star','w','N_real','N_predict','Error_predict'])

## DATA CLEANING
""" Certains retweets ont une magnitude 0.0 (influence nulle) mais contribuent au compte total """
## ITERATE CASCADES
numero = len(indexes)

for cascade_idx in tqdm(range(numero)):
    cascade = cascades.loc[indexes.loc[cascade_idx, 'start_ind']:indexes.loc[cascade_idx, 'end_ind']]
    history = cascade[cascade['time']<=observation_duration]
    if cascade.iloc[-1, 1] >= observation_duration: # cascade de longueur >= 50 retweets, validant la durée d'observation minimale, ne contenant pas un terme de magnitude nulle ou de time nul autre que le 1er
        print(f"fitting params for cascade n°{cascade_idx+1}")
        # Fit hawkes params
        history = history.to_numpy()
        result_params, _ = fit_parameters(history, start_params)
        K_beta_c_theta = dict(K=result_params[0], beta=result_params[1], c=result_params[2], theta=result_params[3])
        prediction = get_total_events(history=history,
                                      T=observation_duration,
                                      params=K_beta_c_theta)
        if prediction['n_star']>=1: # Cas du régime super-critique
        	pass
       	else:
	        RF_features = dict(c = result_params[2],
	                           theta = result_params[3],
	                           A1 = prediction['a1'],
	                           n_star = prediction['n_star'],
	                           w = (len(cascade) - len(history)) * (1 - prediction['n_star']) / prediction['a1'],
	                           N_real = len(cascade),
	                           N_predict = prediction['total'],
	                           Error_predict = 100 * abs(len(cascade) - prediction['total']) / len(cascade))
	        RF_train_data.at[cascade_idx, :] = RF_features

RF_train_data.to_csv('training_data_RF.csv')
##