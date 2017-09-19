"""
Computes and display the number of iterations needed as a function of the gain, 
averaged on several experiment.
"""

# Author: Sebastien Dubois 
#         for ALFA Group, CSAIL, MIT

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

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


def iterationsNeeded(test_name,
                     model_dir, ref_size,
                     first_exp, last_exp,
                     path_file = 'path'):
    """
    Parameters
    ----------
    `test_name` : name of the test instance, should correspond to a 
        folder name in test/

    `model_dir`: a string corresponding to the model and subsize to use,
        written as a path. Ex: 'GCP/5000'

    `ref_size` : the subsampling size to consider as a reference for the 'true' score,
        in general this should be the biggest subsampling size available

    `first_exp` : the index number of the first experiment to take into account

    `last_exp` : the index number of the last experiment to take into account
    
    `path_file` : suffixe for the path file to use.
                  Ex 'path' for param_path.csv, 'opt_path' for param_opt_path.csv, ...

    Returns
    -------
    `all_results` : a list of length 4 with the mean, first quartile, median, 
        third quartile, of the number of iterations needed to reach a given gain.  
        This gain depends on the index i and is : 95 + 0.05*i.
    """

    # TODO : support transformed output to change a bit the parameter path
    # `threshold` : threshold to use to decide whether the difference between 
    #   observations' means is significant, based on Welch's t-test

    # `alpha` : trade-off parameter to compute the score from the significant
    #   mean and the standard deviation. score == m - alpha * std

    # prepare files and folders
    result_path = test_name + "/exp_results/" + model_dir + "/iterations_needed_" + \
                  str(ref_size) + "_" + path_file + "/exp" + str(first_exp) + "_" + str(last_exp) + ".csv"
                  # "_t_" +str(threshold)+"_a_"+str(alpha) +
    result_path_cs = test_name + "/exp_results/" + model_dir + "/cumul_score_" + \
                  str(ref_size) + "_" + path_file + "/exp" + str(first_exp) + "_" + str(last_exp) + ".csv"

    folder = test_name + "/exp_results/" + model_dir + "/iterations_needed_" + str(ref_size) + "_" + path_file
    folder_cs = test_name + "/exp_results/" + model_dir + "/cumul_score_" + str(ref_size) + "_" + path_file

    if not os.path.exists(folder):
        os.mkdir(folder)

    if not os.path.exists(folder_cs):
        os.mkdir(folder_cs)

    if os.path.exists(result_path):
        result = np.genfromtxt(result_path, delimiter = ',')
        return result

    # load 'true' scores
    f = open(test_name + "/scoring_function/" + str(ref_size) + "_output.csv", 'r')
    mean_outputs = []
    for l in f:
        l = l[1:-2]
        string_l = l.split(',')
        all_o = [float(i) for i in string_l]
        mean_outputs.append(np.mean(all_o))
    f.close()
    scores = np.asarray(mean_outputs)

    # load the corresponding parameters
    p_dir = test_name + "/scoring_function/" + str(ref_size) + "_params.csv"
    all_params = np.genfromtxt(p_dir,delimiter=',')

    KNN = NearestNeighbors()
    KNN.fit(all_params)

    m = np.min(scores)
    M = np.max(scores)

    all_iter_needed = []
    all_cumul_score = []

    for n_exp in range(first_exp,last_exp+1):
        path = np.genfromtxt(test_name + "/exp_results/" + model_dir + "/exp" + str(n_exp) + "/param_" + path_file + ".csv",
                             delimiter=',')
        true_score = np.zeros(path.shape[0])
        cumul_score = np.zeros(path.shape[0])
        s = 0
        
        for i in range(path.shape[0]):
            neighbor_idx = KNN.kneighbors(path[i,:],1,return_distance=False)[0]
            s += scores[neighbor_idx]
            true_score[i]  = 100. * (scores[neighbor_idx] - m) / (M - m)
            cumul_score[i] = 100. * ((s/(i+1)) - m) / (M - m)

        n_iter_needed =  np.zeros(101)

        starting_score = 80.
        nb_iter = 0
        for i in range(101):
            while(nb_iter < path.shape[0] and true_score[nb_iter] < starting_score + 0.2*i):
                nb_iter += 1
            if(nb_iter == path.shape[0]):
                print 'Exp is too short'
                nb_iter = 1500
            n_iter_needed[i] = nb_iter

        all_iter_needed.append(n_iter_needed)
        all_cumul_score.append(cumul_score)

    mean_iter_needed   = [np.mean([all_iter_needed[j][i]       for j in range(last_exp+1-first_exp)])      for i in range(101)]
    median_iter_needed = [np.median([all_iter_needed[j][i]     for j in range(last_exp+1-first_exp)])      for i in range(101)]
    q1_iter_needed     = [np.percentile([all_iter_needed[j][i] for j in range(last_exp+1-first_exp)],q=25) for i in range(101)]
    q3_iter_needed     = [np.percentile([all_iter_needed[j][i] for j in range(last_exp+1-first_exp)],q=75) for i in range(101)]

    median_cumul_score = [np.median([all_cumul_score[j][i]     for j in range(last_exp+1-first_exp)])      for i in range((all_cumul_score[0]).shape[0])]

    all_results = np.concatenate((np.atleast_2d(mean_iter_needed),   np.atleast_2d(q1_iter_needed), \
                                  np.atleast_2d(median_iter_needed), np.atleast_2d(q3_iter_needed)))

    np.savetxt(result_path, all_results, delimiter = ',')
    np.savetxt(result_path_cs, np.asarray(median_cumul_score), delimiter = ',')

    return all_results


if __name__ == '__main__':
    first_exp = 1
    last_exp = 50
    test_name = "SentimentAnalysis"
    model_dir = "rand/5000"
    ref_size = 15000
    path_file = "path"

    # threshold = 0.5
    # alpha = 0.5

    mean_iter_needed,q1_iter_needed,median_iter_needed,q3_iter_needed = \
        iterationsNeeded(test_name,
                         model_dir, ref_size,
                         first_exp, last_exp,
                         path_file = path_file)

    abs = 95 + 0.05 * np.asarray(range(101))

    fig = plt.figure(figsize=(15,7))
    plt.plot(abs[median_iter_needed < 1000],median_iter_needed[median_iter_needed < 1000],'c')
    plt.plot(abs[q1_iter_needed < 1000],q1_iter_needed[q1_iter_needed < 1000],'c-.')
    plt.plot(abs[q3_iter_needed < 1000],q3_iter_needed[q3_iter_needed < 1000],'c-.')
    plt.title('Iterations needed')
    plt.xlabel('Percentage of maximum gain')
    plt.ylabel('Number of tested parameters')
    plt.show()