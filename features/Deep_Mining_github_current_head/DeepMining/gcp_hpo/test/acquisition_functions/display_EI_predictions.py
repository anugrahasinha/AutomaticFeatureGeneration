"""
Fits a GCP on some training data of a custom function, and computes the EI 
(Expected Improvement) in each point, as it would be made in Bayezian 
optimization. Also display the results.
"""

# Author: Sebastien Dubois 
#		  for ALFA Group, CSAIL, MIT

# The MIT License (MIT)
# Copyright (c) 2015 Sebastien Dubois

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import numpy as np
import matplotlib.pyplot as plt
import os

import gcp_hpo.search_utils as utils 
from gcp_hpo.gcp.gcp import GaussianCopulaProcess
from gcp_hpo.test.function_utils import artificial_f


def scoring_function(x):
	return artificial_f([x])[0]

def main():
	save_plots = False

	### Set parameters ###
	nugget = 1.e-10
	all_n_clusters = [1,2]
	corr_kernel = 'exponential_periodic'
	GCP_mapWithNoise= False
	sampling_model = 'GCP'
	coef_latent_mapping = 0.1
	prediction_size = 400

	### Set parameters ###
	parameter_bounds = np.asarray( [[0,400]] )
	training_size = 30

	abs = np.atleast_2d(range(0,400)).T
	f_plot = [scoring_function(i) for i in abs[:,0]]

	x_training = []
	y_training = []
	for i in range(training_size):
		x = np.random.uniform(0,400)
		x_training.append(x)
		y_training.append(scoring_function(x))
	x_training = np.atleast_2d(x_training).T
	candidates = abs

	if (save_plots):
		if not os.path.exists('data_EI'):
			os.mkdir('data_EI')

		# store training data
		g=open('data_EI/training_data.csv','w')
		g.write('x,y\n')
		for i in range(training_size):
			g.write( str(x_training[i]) + ',' + str(y_training[i]) + '\n')
		g.close()

	count = 0
	fig = plt.figure()

	for n_clusters in all_n_clusters:
		if (save_plots):
			f=open('data_EI/cluster' + str(n_clusters) +'.csv','w')
			f.write('x,y,pred,ei\n')

		count += 1
		ax = fig.add_subplot(len(all_n_clusters),1,count)
		ax.set_title("GCP prediction")

		gcp = GaussianCopulaProcess(nugget = nugget,
									corr = corr_kernel,
									random_start = 5,
									n_clusters = n_clusters,
		                            coef_latent_mapping = coef_latent_mapping,
								 	mapWithNoise = GCP_mapWithNoise,
					 				useAllNoisyY = False,
					 				model_noise = None,
									try_optimize = True)
		gcp.fit(x_training,y_training)

		print '\nLGCP fitted -', n_clusters, 'clusters'
		print 'Likelihood', np.exp(gcp.reduced_likelihood_function_value_)

		predictions = gcp.predict(candidates,eval_MSE=False,eval_confidence_bounds=False,coef_bound = 1.96,integratedPrediction=False)

		pred,mse = gcp.predict(candidates,eval_MSE=True,transformY=False)
		y_best =np.max(y_training)
		sigma = np.sqrt(mse)
		ei = [ utils.gcp_compute_ei((candidates[i]- gcp.X_mean) / gcp.X_std,pred[i],sigma[i],y_best, \
		                gcp.mapping,gcp.mapping_derivative) \
		        for i in range(candidates.shape[0]) ]
		ei = np.asarray(ei)

		if(save_plots):
			for i in range(abs.shape[0]):
				f.write( str(candidates[i,0]) + ',' + str(f_plot[i]) +',' + str(predictions[i]) + ',' + str(ei[i]) +'\n' )
			f.close()

		ax.plot(abs,f_plot)
		ax.plot(candidates,predictions,'r+',label='GCP predictions')
		ax.plot(x_training,y_training,'bo',label='Training points')
		ax.plot(candidates,100.*ei,'g+',label='EI')

	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()