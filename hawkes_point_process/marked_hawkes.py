import numpy as np
from functools import partial
from typing import Dict
import ipopt

default_params = dict(
	K=0.8,
	beta=0.6,
	c=10,
	theta=0.8
)


def kernel_fct(event, t, params=default_params, alpha = 2.016, mmin = 1, inclusive = True):
	K, beta, c, theta = params['K'], params['beta'], params['c'], params['theta']
	mi = event[0]
	ti = event[1]
	val_i = np.zeros(len(t))
	if not inclusive and mi == mmin:
		return val_i
	if mi >= mmin:
		# Virality * Influence of the user * Decaying (relaxation kernel)
		val_i[t>=ti] = K * (mi / mmin)**beta / (t[t>=ti] - ti + c)**(1+theta)
	if not inclusive:
		val_i[t==ti] = 0

	return val_i


# def lambda_rate_v1(t, history, params, inclusive = False):
# 	K, beta, c, theta = params['K'], params['beta'], params['c'], params['theta']
# 	res = np.zeros((len(t),))
# 	for i in range(len(t)):
# 		history_iter = history[history['time']<=t[i]]
# 		for event_idx in range(len(history_iter)):
# 			res[i] += kernel_fct(event=history_iter.iloc[event_idx].values,
# 								  t=np.array([t[i]]),
# 								  params=params, inclusive=inclusive)[0]
# 	return res


def lambda_rate(t: np.array, history: np.ndarray, params: Dict[str, float], inclusive = False):
	K, beta, c, theta = params['K'], params['beta'], params['c'], params['theta']
	res = np.zeros((len(t),))
	for i in range(len(t)):
		history_iter = history[history[:, 1]<=t[i]]
		for event_idx in range(len(history_iter)):
			res[i] += kernel_fct(event=history_iter[event_idx],
								 t=np.array([t[i]]),
								 params=params, inclusive=inclusive)[0]
	return res


# def integrate_lambda_v1(upper, history, params, mmin = 1):
# 	""" calculate integral of lambda from 0 to upper """

# 	K, beta, c, theta = params['K'], params['beta'], params['c'], params['theta']
# 	return K * sum(history.apply(
# 			lambda event: (event['magnitude'] / mmin)**beta * (1 / (theta * c**theta) - \
# 				1 / (theta * (upper + c - event['time'])**theta)),
# 		axis=1).to_numpy())

def integrate_lambda(upper: float, history: np.ndarray, params: Dict[str, float], mmin=1):
	""" calculate integral of lambda from 0 to upper """
	K, beta, c, theta = params['K'], params['beta'], params['c'], params['theta']
	return K * ((history[:, 0] / mmin)**beta * (1 / (theta * c**theta) - \
		   1 / (theta * (upper + c - history[:, 1])**theta))).sum()


# def neg_log_likelihood_v1(history, params):
# 	""" calculates the negative log-likelihood.
# 		Minimizing the -1 * log-likelihood amounts to maximizing the log-likehood """

# 	T = history['time'].max()
# 	return integrate_lambda(T, history, params) - \
# 		sum(np.log(lambda_rate(history.loc[1:, 'time'].to_numpy(), history, params)))

def neg_log_likelihood(history: np.ndarray, params: Dict[str, float]):
	""" calculates the negative log-likelihood.
		Minimizing the -1 * log-likelihood amounts to maximizing the log-likehood """

	T = np.amax(history[:, 1])
	return integrate_lambda(T, history, params) - \
		sum(np.log(lambda_rate(history[1:, 1], history, params))) 


def closed_gradient(history: np.ndarray, params: Dict[str, float]):
	""" calculate the dervative in closed form for our Power Law Social Kernel """

	T = np.amax(history[:, 1])
	n = history.shape[0]
	K, beta, c, theta = params['K'], params['beta'], params['c'], params['theta']


	# In all the following calculation, res is the part of derivative coming 
	# from bigLambda part and first is derivative coming from 
	# summation of log(lambda(ti))

	### calculating derivative wrt k in closed form
	mi_beta = history[:, 0] ** beta
	first_part = 1 / (c ** theta)
	second_part = T + c - history[:, 1]
	second_part_k = 1 / (second_part ** theta)
	res = (mi_beta * (first_part - second_part_k)).sum() / theta

	deriv_K = ((n-1) / K) - res

	### calculating dervative wrt beta in closed form
	## MAR: second part is almost identical to the one above, reusing.
	res = K * (mi_beta * np.log(history[:, 0]) * (first_part - second_part_k)).sum() / theta

	# we go only from second row as first row has no history before
	# (inner sum goes strictly one less) and we do not have a mu(background rate)
	# numerically (history$magnitude[:i-1] or (history$time[:i-1] is zero
	first = dict(first=[], denominator=[])
	for i in range(1, n):
		## MAR: nominator and denominator are basically the same, just nominator has an additional log(m_j)
		denominator = ((history[i, 1] + c - history[:i, 1])**(-1-theta)) * \
			(history[:i, 0]**beta)
		numerator = denominator * np.log(history[:i, 0])
		first['first'].append(numerator.sum() / denominator.sum())
		first['denominator'].append(denominator)

	deriv_Beta = sum(first['first']) - res

	### calculating derivative wrt c in closed form
	first_part = 1 / (c**(1+theta))
	second_part_c = 1 / (second_part**(1+theta))
	res = (mi_beta * (second_part_c - first_part)).sum() * K

	first_c = 0
	for i in range(1, n):
		denominator = first['denominator'][i-1]
		numerator = (((history[i, 1] + c - history[:i, 1])**(-2-theta)) * \
			 (-1-theta) * (history[:i, 0]**beta)).sum()
		first_c += numerator / sum(denominator)

	deriv_C = first_c - res

	#### calculating derivative wrt theta in closed form
	first_part = second_part**(-theta) * (theta * np.log(second_part) + 1)
	second_part_theta = c**(-theta) * ((theta * np.log(c)) + 1)
	res1 = (mi_beta * (first_part - second_part_theta)).sum() * K / (theta**2)

	res = 0
	for i in range(n):
		# calculating in three parts so that can be debugged easily and 
	    # expressed in terms of smaller terms
	    mi_beta = history[i, 0]**beta
	    first_part = (T + c - history[i, 1])**(-theta) * \
	    	(theta * np.log(T + c - history[i, 1]) + 1)
	    second_part = c**(-theta) * (theta * np.log(c) + 1)
	    res += mi_beta * (first_part - second_part)
	res *= K / (theta**2)

	first_theta = 0
	for i in range(1, n):
		denominator = first['denominator'][i-1]
		numerator = np.log(history[i, 1] + c - history[:i, 1]) * denominator
		first_theta += sum(numerator) / sum(denominator)
	first_theta *= -1

	deriv_Theta = first_theta - res

	## multiply by -1 because we are minimizing log-likelihood
	return np.array([-deriv_K, -deriv_Beta, -deriv_C, -deriv_Theta])

# def closed_gradient_v1(history, params):
# 	""" calculate the dervative in closed form for our Power Law Social Kernel """
# 	T = history['time'].max()
# 	n = len(history)
# 	K, beta, c, theta = params['K'], params['beta'], params['c'], params['theta']

# 	# In all the following calculation, res is the part of derivative coming 
# 	# from bigLambda part and first is derivative coming from 
# 	# summation of log(lambda(ti))

# 	### calculating derivative wrt k in closed form
# 	mi_beta = history['magnitude'] ** beta
# 	first_part = 1 / (c ** theta)
# 	second_part = T + c - history['time']
# 	second_part_k = 1 / (second_part ** theta)
# 	res = (mi_beta * (first_part - second_part_k)).sum() / theta
# 	deriv_K = ((n-1) / K) - res

# 	### calculating dervative wrt beta in closed form
# 	## MAR: second part is almost identical to the one above, reusing.
# 	res = K * (mi_beta * np.log(history['magnitude']) * (first_part - second_part_k)).sum() / theta

# 	# we go only from second row as first row has no history before
# 	# (inner sum goes strictly one less) and we do not have a mu(background rate)
# 	# numerically (history$magnitude[:i-1] or (history$time[:i-1] is zero
# 	first = dict(first=[], denominator=[])
# 	for i in range(1, n):
# 		## MAR: nominator and denominator are basically the same, just nominator has an additional log(m_j)
# 		denominator = ((history.loc[i, 'time'] + c - history.loc[:i-1, 'time'])**(-1-theta)) * \
# 			(history.loc[:i-1, 'magnitude']**beta)
# 		numerator = denominator * np.log(history.loc[:i-1, 'magnitude'])
# 		first['first'].append(numerator.sum() / denominator.sum())
# 		first['denominator'].append(denominator.to_numpy())

# 	deriv_Beta = sum(first['first']) - res
	
# 	### calculating derivative wrt c in closed form
# 	first_part = 1 / (c**(1+theta))
# 	second_part_c = 1 / (second_part**(1+theta))
# 	res = (mi_beta * (second_part_c - first_part)).sum() * K

# 	first_c = 0
# 	for i in range(1, n):
# 		denominator = first['denominator'][i-1]
# 		numerator = (((history.loc[i, 'time'] + c - history.loc[:i-1, 'time'])**(-2-theta)) * \
# 			 (-1-theta) * (history.loc[:i-1, 'magnitude']**beta)).sum()
# 		first_c += numerator / sum(denominator)

# 	deriv_C = first_c - res

# 	#### calculating derivative wrt theta in closed form
# 	first_part = second_part**(-theta) * (theta * np.log(second_part) + 1)
# 	second_part_theta = c**(-theta) * ((theta * np.log(c)) + 1)
# 	res1 = (mi_beta * (first_part - second_part_theta)).sum() * K / (theta**2)

# 	res = 0
# 	for i in range(n):
# 		# calculating in three parts so that can be debugged easily and 
# 	    # expressed in terms of smaller terms
# 	    mi_beta = history.loc[i, 'magnitude']**beta
# 	    first_part = (T + c - history.loc[i, 'time'])**(-theta) * \
# 	    	(theta * np.log(T + c - history.loc[i, 'time']) + 1)
# 	    second_part = c**(-theta) * (theta * np.log(c) + 1)
# 	    res += mi_beta * (first_part - second_part)
# 	res *= K / (theta**2)

# 	first_theta = 0
# 	for i in range(1, n):
# 		denominator = first['denominator'][i-1]
# 		numerator = np.log(history.loc[i, 'time'] + c - history.loc[:i-1, 'time']).to_numpy() * denominator
# 		first_theta += sum(numerator) / sum(denominator)
# 	first_theta *= -1

# 	deriv_Theta = first_theta - res

# 	## multiply by -1 because we are minimizing log-likelihood
# 	return np.array([-deriv_K, -deriv_Beta, -deriv_C, -deriv_Theta])


def constraint(params: Dict[str, float]):
	""" specify the constraint for Power Law Kernel """
	K, beta, c, theta = params['K'], params['beta'], params['c'], params['theta']
	return np.log(K) + np.log(1.1016) - np.log(1.016 - beta) - \
		np.log(theta) - (theta * np.log(c))


def jacobian(params: Dict[str, float]):
	""" specify the Jacobian for Power Law Kernel """
	K, beta, c, theta = params['K'], params['beta'], params['c'], params['theta']
	return np.array([1 / K, 1 / (1.016 - beta), -theta / c, -(1 / theta) - np.log(c)])


def create_start_points(seed=None):
	"""  create random and defined starting points """
	np.random.seed(seed)
	start_K = np.random.uniform(np.finfo(float).eps, 1, 5)
	start_beta = np.random.uniform(np.finfo(float).eps, 1.016, 5)
	start_theta = np.random.uniform(np.finfo(float).eps, 1, 5)
	start_c = np.random.uniform(np.finfo(float).eps, 1, 5)
	start_K = np.concatenate((start_K, np.array([0.1, 0.5, 0.01])))
	start_beta = np.concatenate((start_K, np.array([0.1, 0.5, 0.01])))
	start_theta = np.concatenate((start_K, np.array([0.1, 0.5, 0.01])))
	start_c = np.concatenate((start_K, np.array([0.1, 0.5, 0.01])))
	start = dict(K=start_K, beta=start_beta, c=start_c, theta=start_theta)
	return start


def fit_parameters(history):
	""" run ipopt to fit hawkes parameters from data in history """
	x0 = np.array([1, 1, 250, 1])
	cl = np.array([np.log(np.finfo(float).eps)]) # lower bounds on constraints
	cu = np.array([np.log(1 - np.finfo(float).eps)]) # upper bounds on constraints
	lb = np.array([0, 0, 0, 0]) # lower bounds on variables
	ub = np.array([1, 1.016, float('inf'), float('inf')]) # upper bounds on variables

	nlp = ipopt.problem(n=len(x0),
						m=len(cl),
						problem_obj=hawkes_solver(history),
						lb=lb,
			            ub=ub,
			            cl=cl,
			            cu=cu)

	nlp.addOption('print_level', 0)
	nlp.addOption('max_iter', 10000)

	x, info = nlp.solve(x0)
	return x, info



class hawkes_solver(object):
	def __init__(self, history: np.ndarray):
		self.history = history

	def objective(self, x):
		""" callback for calculating the objective """
		params = dict(K=x[0], beta=x[1], c=x[2], theta=x[3])
		return neg_log_likelihood(self.history, params)

	def gradient(self, x):
		""" callback for calculating the gradient """
		params = dict(K=x[0], beta=x[1], c=x[2], theta=x[3])
		return closed_gradient(self.history, params)

	def constraints(self, x):
		""" callback for calculating the constraints """
		params = dict(K=x[0], beta=x[1], c=x[2], theta=x[3])
		return constraint(params)

	def jacobian(self, x):
		""" callback for calculating the Jacobian """
		params = dict(K=x[0], beta=x[1], c=x[2], theta=x[3])
		return jacobian(params)

	def intermediate(
			self,
			alg_mod,
			iter_count,
			obj_value,
			inf_pr,
			inf_du,
			mu,
			d_norm,
			regularization_size,
			alpha_du,
			alpha_pr,
			ls_trials):
		print("Objective value at iteration #%d is - %g" % (iter_count, obj_value))


import pandas as pd
real_cascade = pd.read_csv('example_book.csv', names=['magnitude', 'time'], header=0)
real_cascade.index -= 1
observation_duration = 600 # seconds
history = real_cascade[real_cascade['time']<=observation_duration]

x, info = fit_parameters(history.to_numpy())
print(x)
print('####')
print(info)
