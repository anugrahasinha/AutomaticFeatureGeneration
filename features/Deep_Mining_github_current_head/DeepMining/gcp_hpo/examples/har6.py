"""
Testing SmartSearch on the Hartmann 6D function.  

The Hartmann 6D function is a classic example to evaluate nonlinear optimization algorithms.  
"""

from gcp_hpo.smart_search import SmartSearch
import numpy as np

def har6(x):
    """6d Hartmann test function
    constraints:
    0 <= xi <= 1, i = 1..6
    global optimum at (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573),
    where har6 = 3.32236"""


    a = np.array([[10.0,   3.0, 17.0,   3.5,  1.7,  8.0],
                [ 0.05, 10.0, 17.0,   0.1,  8.0, 14.0],
                [ 3.0,   3.5,  1.7,  10.0, 17.0,  8.0],
                [17.0,   8.0,  0.05, 10.0,  0.1, 14.0]])
    c = np.array([1.0, 1.2, 3.0, 3.2])
    p = np.array([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
                [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])
    s = 0
    for i in [0,1,2,3]:
        sm = a[i,0]*(x[0]-p[i,0])**2
        sm += a[i,1]*(x[1]-p[i,1])**2
        sm += a[i,2]*(x[2]-p[i,2])**2
        sm += a[i,3]*(x[3]-p[i,3])**2
        sm += a[i,4]*(x[4]-p[i,4])**2
        sm += a[i,5]*(x[5]-p[i,5])**2
        s += c[i]*np.exp(-sm)
    
    return [s]

def scoring_function(p_dict):
	p_vector = [p_dict['a'],
				p_dict['b'],
				p_dict['c'],
				p_dict['d'],
				p_dict['e'],
				p_dict['f'] ]
	return har6(p_vector)

def main():
	### Set parameters ###
	parameters = { 'a' : ['float',[0,1]],
				   'b' : ['float',[0,1]],
				   'c' : ['float',[0,1]],
				   'd' : ['float',[0,1]],
				   'e' : ['float',[0,1]],
				   'f' : ['float',[0,1]] }
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