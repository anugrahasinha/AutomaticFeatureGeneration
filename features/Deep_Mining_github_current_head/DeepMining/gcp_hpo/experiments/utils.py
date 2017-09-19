"""
Utilities to analyze the results of HPO experiments.
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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import six
from gcp_hpo.experiments.iterations_needed import iterationsNeeded


def load_data(subsize, dir_ = ''):
    mean_outputs = []
    raw_outputs = []

    f = open(dir_ + "scoring_function/" + str(subsize) + "_output.csv",'r')

    for l in f:
        l = l[1:-2]
        string_l = l.split(',')
        all_o = [float(i) for i in string_l]
        raw_outputs.append(all_o)
        mean_outputs.append(np.mean(all_o))
    f.close()

    params = np.genfromtxt((dir_ + 'scoring_function/' + str(subsize) + '_params.csv'), 
                            delimiter=',')

    return raw_outputs, mean_outputs, params


def load_output(dir_):
    mean_outputs = []
    std_outputs = []
    f = open(dir_ + "/output.csv",'r')

    for l in f:
        l = l[1:-2]
        string_l = l.split(',')
        all_o = [float(i) for i in string_l]
        mean_outputs.append(np.mean(all_o))
        std_outputs.append(np.std(all_o))
    f.close()

    return mean_outputs, std_outputs


def score_distribution(subsize, dir_ = ''):
    raw_outputs, mean_outputs, params = load_data(subsize, dir_)
    plt.figure()
    plt.hist(mean_outputs, bins = 100)
    plt.title("Subsize == " + str(subsize))
    plt.show()


def show_iter_needed(test_name, first_exp, last_exp, models, ref_size, path_file = "path"):
    colors_ = list(six.iteritems(colors.cnames))

    fig = plt.figure(figsize=(15,7))
    abs = 95 + 0.05 * np.asarray(range(101))

    c = 0
    for model_dir in models:
        mean_iter_needed,q1_iter_needed,median_iter_needed,q3_iter_needed = \
            iterationsNeeded(test_name,
                             model_dir, ref_size,
                             first_exp, last_exp,
                             path_file = path_file)

        plt.plot(abs[median_iter_needed < 1000],median_iter_needed[median_iter_needed < 1000], color = colors_[c][0])
        # plt.plot(abs[q1_iter_needed < 1000],q1_iter_needed[q1_iter_needed < 1000], marker = '-.')
        # plt.plot(abs[q3_iter_needed < 1000],q3_iter_needed[q3_iter_needed < 1000], marker = '-.')
        c += 1

    plt.title('Iterations needed')
    plt.xlabel('Percentage of maximum gain')
    plt.ylabel('Number of tested parameters')
    plt.show()


def show_cumul_score(test_name, first_exp, last_exp, models, ref_size, path_file = "path"):
    colors_ = list(six.iteritems(colors.cnames))

    fig = plt.figure(figsize=(15,7))
    abs = 95 + 0.05 * np.asarray(range(101))

    c = 0
    for model_dir in models:
        path = test_name + "/exp_results/" + model_dir + "/cumul_score_" + str(ref_size) + \
               "_" + path_file + "/exp" + str(first_exp) + "_" + str(last_exp) + ".csv"
        
        median_cumul_score = np.genfromtxt(path, delimiter = ',')
        n_steps = median_cumul_score.shape[0]
        
        plt.plot(range(n_steps), median_cumul_score, color = colors_[c][0])
        
        c += 1

    plt.title('Median of the cumulative gain')
    plt.xlabel('Step')
    plt.ylabel('Median gain over all runs')
    plt.show()