import pandas as pd
import numpy as np
from tqdm import tqdm

from hawkes_point_process.marked_hawkes import create_start_points, fit_parameters, get_total_events


OBSERVATION_TIME = 3600
SEED = 0

# create starting params for fitting hawkes process
start_params = np.array([np.random.choice(param) for param in create_start_points(SEED).values()])

# read tweets data
tweets_dir = './data'
data_df = pd.read_csv(tweets_dir+'/data.csv')
data_df = data_df[['magnitude', 'time']]
index_df = pd.read_csv(tweets_dir+'/index.csv')
index_df = index_df.iloc[:50] # fit params for first 50 cascades

# create a fataframe that will contain real and predicted final cascade sizes
cascade_sizes_df = pd.DataFrame()
cascade_sizes_df['real_size'] = index_df.apply(lambda row: row['end_ind'] - row['start_ind'] + 1, axis=1)

num_cascades = len(index_df)
predicted_cascades_len = np.zeros((num_cascades))
for cascade_idx in tqdm(range(num_cascades)):
	cascade = data_df.iloc[index_df.loc[cascade_idx, 'start_ind']-1:index_df.loc[cascade_idx, 'end_ind']]

	# fit hawkes params
	print(f"fitting params for cascade n°{cascade_idx+1}")
	history = cascade[cascade['time']<=OBSERVATION_TIME].to_numpy()
	result_params, _ = fit_parameters(history, start_params)
	fitted_params = dict(K=result_params[0], beta=result_params[1], c=result_params[2], theta=result_params[3])

	# predict total cascade length
	prediction = get_total_events(history=history, T=OBSERVATION_TIME, params=fitted_params)
	predicted_cascades_len[cascade_idx] = prediction['total']
	
cascade_sizes_df['pred_size'] = predicted_cascades_len
cascade_sizes_df['pred_error'] = 100 * abs(cascade_sizes_df['real_size'] - cascade_sizes_df['pred_size']) / cascade_sizes_df['real_size']

cascade_sizes_df.to_csv('cascades_size_pred_example.csv')

print(cascade_sizes_df)
