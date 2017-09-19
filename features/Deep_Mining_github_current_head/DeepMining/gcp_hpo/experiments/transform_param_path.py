"""
Transform the `param_path.csv` file into a new path were the hyper-parameter at
step k is the best guess with the data available up to this step.
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
from gcp_hpo.experiments.utils import load_output

def transformPath(test_name,
                  model_dir,
                  first_exp, last_exp,
                  alpha = 1.,
                  path_file = 'opt_path'):
    """
    Parameters
    ----------
    `test_name` : name of the test instance, should correspond to a 
        folder name in test/

    `model_dir`: a string corresponding to the model and subsize to use,
        written as a path. Ex: 'GCP/5000'

    `first_exp` : the index number of the first experiment to take into account

    `last_exp` : the index number of the last experiment to take into account

    `alpha` : the quality of a CV result is defined as mean(CV) - alpha * std(CV)

    `path_file` : suffixe for the path file to use.
                  Ex:'opt_path' for param_opt_path.csv
    """

    for n_exp in range(first_exp,last_exp+1):
        dir_ = test_name + "/exp_results/" + model_dir + "/exp" + str(n_exp)
        path = np.genfromtxt(dir_ + "/param_path.csv",
                             delimiter=',')
        mean_outputs, std_outputs = load_output(dir_)
        scores = [mean_outputs[i] - alpha * std_outputs[i] for i in range(len(mean_outputs))]
        scores = np.asarray(scores)
        
        opt_path = np.asarray([path[np.argmax(scores[:(k+1)]), :] for k in range(scores.shape[0])])
        
        np.savetxt(dir_ + "/param_" + path_file + ".csv", opt_path, delimiter = ',')


if __name__ == '__main__':
    first_exp = 1
    last_exp = 50
    test_name = "SentimentAnalysis"
    model_dir = "rand/5000"
    # alpha = 1

    transformPath(test_name, model_dir,
                  first_exp, last_exp)