#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

sys.path.append('../')

from hawkes_point_process.marked_hawkes import create_start_points, fit_parameters, get_total_events


## COMPLETE DATA HISTORY
Indexes = pd.read_csv('index.csv', names=['start_ind', 'end_ind'], header=0)
cascades = pd.read_csv('data.csv', names=['time', 'magnitude'], header=0)
cascades = cascades[['magnitude','time']]
cascades.index += 1 # Match with the index.csv values


# =============================================================================
# MANUAL INTERVENTION
# =============================================================================
"""
Long process (1h for 60 cascades with laptop performance)
Hence, apply on batches of 20 cascades before saving fitted parameters to .csv
So process can be killed without a big loss
"""
index_start, index_length, occurences = 3000, 20, 35


## CONTEXT PARAMETERS
observation_duration = 600 # seconds = 10 mins
start_params = np.array([np.random.choice(param) for param in create_start_points(0).values()]) #np.array([.8, .6, 100, .8])

## TRAINING DATA FOR RANDOM FOREST
RF_train_data = pd.read_csv('training_data_RF.csv', names=['c','theta','A1','n_star','w','N_real','N_predict','Error_predict','start_ind'], header=0)
#RF_train_data = pd.DataFrame(columns=['c','theta','A1','n_star','w','N_real','N_predict','Error_predict'])


## DATA CLEANING
""" Certains retweets ont une magnitude 0.0 (influence nulle) mais contribuent au compte total """

## ITERATE CASCADES
for occ in range(occurences):
    indexes = Indexes.iloc[index_start+occ*index_length:index_start+(occ+1)*index_length]
    numero = len(indexes)
    print("########### OCURRENCE : ", occ, "\n\n\n")
    for cascade_idx in tqdm(range(numero)):
        cascade = cascades.loc[indexes.iloc[cascade_idx, 0]:indexes.iloc[cascade_idx, 1]]
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
        	                           Error_predict = 100 * abs(len(cascade) - prediction['total']) / len(cascade),
                                    start_ind = indexes.iloc[cascade_idx, 0])
        	        RF_train_data.at[index_start+occ*index_length+cascade_idx, :] = RF_features
    RF_train_data.to_csv('training_data_RF.csv', index=False)
    RF_train_data = pd.read_csv('training_data_RF.csv', names=['c','theta','A1','n_star','w','N_real','N_predict','Error_predict','start_ind'], header=0)
##