# Test

The `gcp_hpo.test` module contains some scripts to test how GCP behaves.  

### Regression
The regression folder tests GCP on regression tasks. While `display_branin_function.py` and `simple_test.py` are mostly there to visualize what is happening, 
`full_test.py` can be used to quantify its accuracy. More precisely, this enables to test GCP for regression on different functions (see the file `function_utils.py`) 
and to measure the likelihood of the fit as well as the mean squared errors of the predictions.

### Acquisition functions
In GCP-based Bayezian optimization, a GCP is fitted on an unknown function and the fit is used to compute some acquisition function. The folder `acquisition_functions` folder 
contains two scripts to see the behaviors of the Expected Improvement and the confidence bounds with GCP. 