"""
Testing SmartSearch on the Braning function.  

The Branin function is a classic example to evaluate nonlinear optimization algorithms.  
The goal is to find the minimum, we use here the opposite of the Branin function to match
the maximization process.
"""
# x in [-5,10] -  y in [0,15]
# min is 0.398

from gcp_hpo.smart_search import SmartSearch
import numpy as np
import math

def branin(x, y):
	"""
	The opposite of the Branin function.
	"""
	result = np.square(y - (5.1/(4*np.square(math.pi)))*np.square(x) + \
		(5/math.pi)*x - 6) + 10*(1-(1./(8*math.pi)))*np.cos(x) + 10
	return [-result]


def noisy_branin(x, y):
	"""
	The opposite of the Branin function with an additive Gaussian noise.
	"""
	result = np.square(y - (5.1/(4*np.square(math.pi)))*np.square(x) + \
		(5/math.pi)*x - 6) + 10*(1-(1./(8*math.pi)))*np.cos(x) + 10
	result += np.random.normal() * 50.
	return [-result]


def scoring_function(p_dict):
	x,y = p_dict['x'], p_dict['y']
	x = x -5.
	y= y
	return branin(x,y)


def main():
	### Set parameters ###
	parameters = { 'x' : ['float',[0,15]],
				   'y' : ['float',[0,15]] }
	nugget = 1.e-10
	n_clusters = 1
	cluster_evol ='constant'
	corr_kernel = 'squared_exponential'
	mapWithNoise= False
	model_noise = None
	sampling_model = 'GCP'
	n_candidates= 300
	n_random_init= 15
	n_iter = 100
	nb_iter_final = 0
	acquisition_function = 'UCB'

	search = SmartSearch(parameters,
				estimator=scoring_function,
				corr_kernel = corr_kernel,
				acquisition_function = acquisition_function,
				GCP_mapWithNoise=mapWithNoise,
				model_noise = model_noise,
				model = sampling_model, 
				n_candidates=n_candidates,
				n_iter = n_iter,
				n_init = n_random_init,
				n_final_iter=nb_iter_final,
				n_clusters=n_clusters, 
				cluster_evol = cluster_evol,
				verbose=2,
				detailed_res = 0)

	search._fit()


if __name__ == '__main__':
	main()