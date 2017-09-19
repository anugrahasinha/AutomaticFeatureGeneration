# Experiments

The `gcp_hpo.experiments` module contains data and scripts that can be used to test GCP-based hyper-parameter optimization. Designed for research purposes, 
this module is relevant when one wants to run several hyper-parameter optimization process with different configurations, in order to compare them for example. That is why this 
module relies on two different components: off-line and on-line computations.  

### Off-line computations
First, train and test a pipeline for many parameters, and store their performances in the folder `test_name/scoring_function`. This can be done, for example, by running `SmartSearch` 
but with randomized search. 

### On-line computations
Simulate as many hyper-parameter optimization processes as you want, eventually with different configurations, with the script `run_experiment`. There, instead of training/testing 
a pipeline, the performances of a parameter will actually be a query in the database built from the off-line computations.

### Transform the results
Run `transform_param_path` to convert the path of the tested hyper-parameters into the series of the best guess at each step, to simulate the hyper-parameter that
would have been selected with a given budget of computations.  

Then run `iterations_needed` to see how many parameters should be tested to reach a given gain, ie how a SmartSearch configuration performs on this test instance. When doing so,
you can choose the file to use for the hyper-parameter path. This will also compute the `cumul_score`, which is the averaged true score of all the hyperparameters visited up to each step,
which should provide an idea of the overall quality of the hyperparameters tested.

### Files and directory structure  
Each test instance follows the same directory structure, and all files are in the folder `experiments/test_name`:  
- `config.yml` : a yaml file to set the parameters used to run `SmartSearch`  
- `scoring_function/` : data from off-line computations. `subsize_params.csv` contains the parameters tested, and `subsize_output.csv` the raw outputs given by the scoring function 
(all the cross-validation estimations) on sub-sampled datasets of size `subsize`.   
- `exp_results/model/subsize/expXXX/` : data returned by `runExperiment`, where XXX is the number set in the config file, `subsize` is the sub-sample size used
by SmartSearch, and `model` is either `rand`, `GP`, or `GCP`.  
- `exp_results/model/subsize/iterations_needed_refSize_pathFile/expXXX_YYY` : the mean, median, first and third quartiles of the iterations needed to reach a given score gain, over experiments 
XXX to YYY, using the dataset of size `refSize` to set the 'true scores', and the file `pathFile` for the parameter path.  
- `exp_results/model/subsize/cumul_score_refSize_pathFile/expXXX_YYY` : the median of the averaged gain of all hyperparameters tested up to each iteration, over experiments 
XXX to YYY, using the dataset of size `refSize` to set the 'true scores', and the file `pathFile` for the parameter path.  