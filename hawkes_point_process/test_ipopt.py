#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

sys.path.append('../')

from hawkes_point_process.marked_hawkes import create_start_points, fit_parameters, get_total_events


## COMPLETE DATA HISTORY
Indexes = pd.read_csv("../data/tweets/index.csv", names=['start_ind', 'end_ind'], header=0)
cascades = pd.read_csv("../data/tweets/data.csv", names=['time', 'magnitude'], header=0)
cascades = cascades[['magnitude','time']]
cascades.index += 1 # Match with the index.csv values


# =============================================================================
# EXEMPLE TO TEST PROGRAM WITH n SAMPLES
# =============================================================================
indexes = Indexes.sample(n=7)


## CONTEXT PARAMETERS
observation_duration = 600 # seconds = 10 mins
start_params = np.array([np.random.choice(param) for param in create_start_points(0).values()]) #np.array([.8, .6, 100, .8])


numero = len(indexes)
for cascade_idx in tqdm(range(numero)):
    cascade = cascades.loc[indexes.iloc[cascade_idx, 0]:indexes.iloc[cascade_idx, 1]]
    history = cascade[cascade['time']<=observation_duration]
    if cascade.iloc[-1, 1] >= observation_duration: # cascade de longueur >= 50 retweets, validant la durée d'observation minimale, ne contenant pas un terme de magnitude nulle ou de time nul autre que le 1er
        print("#########################################")
        print('\n')
        print(f"fitting params for cascade n°{cascade_idx+1}")
        # Fit hawkes params
        history = history.to_numpy()
        result_params, _ = fit_parameters(history, start_params)
        K_beta_c_theta = dict(K=result_params[0], beta=result_params[1], c=result_params[2], theta=result_params[3])
        prediction = get_total_events(history=history,
                                      T=observation_duration,
                                      params=K_beta_c_theta)
        print("_________________________________________")
        print(f"Observed length before observation time : {len(history)}; real cascade length : {len(cascade)}")
        print("_________________________________________")
        print(f"Fitted params : K = {result_params[0]}; beta = {result_params[1]}; c = {result_params[2]}; theta = {result_params[3]}") 
        print("_________________________________________")
        print(f"Generative model params : n* = {prediction['n_star']}; A1 = {prediction['a1']}; N_pred = n + A1 / (1-n*) = {prediction['total']}")
        print("_________________________________________")
##