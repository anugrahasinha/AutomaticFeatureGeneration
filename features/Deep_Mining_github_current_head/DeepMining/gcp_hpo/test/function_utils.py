"""
Defines functions to use for test purposes.
"""

# Author: Sebastien Dubois 
#     for ALFA Group, CSAIL, MIT

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
import math
import os
import gcp_hpo.experiments
from sklearn.neighbors import NearestNeighbors

dir_ = os.path.dirname(gcp_hpo.experiments.__file__)

def artificial_f(x):
	x = x[0]
	res = (70-7*np.exp(x/50. - ((x-55.)**2)/500.) + 6*np.sin(x/40.) +3./(1.1+np.cos(x/50.)) - 15./(3.3-3*np.sin((x-70)/25.)))/100.
	return [res]

def branin_f(p_vector):
	x,y = p_vector
	x = x -5.
	y= y
	result = np.square(y - (5.1/(4*np.square(math.pi)))*np.square(x) + \
		(5/math.pi)*x - 6) + 10*(1-(1./(8*math.pi)))*np.cos(x) + 10
	return [-result]

def har6(x):
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

# MNIST data
mnist_output = []
f =open((dir_ + "/MNIST/scoring_function/output.csv"),'r')
for l in f:
  l = l[1:-3]
  string_l = l.split(',')
  mnist_output.append( [ float(i) for i in string_l] )
f.close()
mnist_params = np.genfromtxt((dir_ + "/MNIST/scoring_function/params.csv"),delimiter=',')
mnist_KNN = NearestNeighbors()
mnist_KNN.fit(mnist_params)

# Popcorn data
popcorn_output = []
f =open((dir_ + "/Bags_of_Popcorn/scoring_function/output.csv"),'r')
for l in f:
  l = l[1:-3]
  string_l = l.split(',')
  popcorn_output.append( [ float(i) for i in string_l] )
f.close()
popcorn_params = np.genfromtxt((dir_ + "/Bags_of_Popcorn/scoring_function/params.csv"),delimiter=',')
popcorn_KNN = NearestNeighbors()
popcorn_KNN.fit(popcorn_params)

# function that retrieves a performance evaluation from the stored results
def mnist_f(p):
	idx = mnist_KNN.kneighbors(p,1,return_distance=False)[0]
	all_o = mnist_output[idx]
	# r = np.random.randint(len(all_o)/5)
	# return all_o[(5*r):(5*r+5)]
	return [ np.mean(all_o)]

def popcorn_f(p):
	idx = popcorn_KNN.kneighbors(p,1,return_distance=False)[0]
	all_o = popcorn_output[idx]
	return [ np.mean(all_o)]