## Copula-based Hyper-parameter Optimization for Machine Learning Pipelines ##

This repository contains all the code implementing the **Gaussian Copula Process (GCP)** and a **hyper-parameter optimization** technique based on it.  
All the code is in Python and mainly uses Numpy, Scipy and Scikit-Learn.


### Getting started 
#### - `gcp_hpo.examples`
One can easily run a GCP-based hyper-parameter optimization process thanks to this code. This is mostly done by the **SmartSearch** object, which iteratively ask to assess the quality of a selected hyper-parameter set. This quality should be returned by the **scoring function** which is implemented by the user and depends on the pipeline. This function should return a list of performance estimations, which would usually be either a single estimation or all k-fold cross-validation results. **SmartSearch** also handles Scikit-Learn pipelines interface so that it is really easy to run the hyper-parameter optimization. 
Check the examples for details on how to use SmartSearch.


### Real-world examples for hyper-parameter optimization research 
#### - `gcp_hpo.experiments`
Two real examples are included in the repository: the **Sentiment Analysis problem** for IMDB reviews (cf. [Kaggle's competition](https://www.kaggle.com/c/word2vec-nlp-tutorial)) in folder `experiments/SentimentAnalysis`, and the **Handwritten digits** one from the MNIST database (cf. [Kaggle's competition](https://www.kaggle.com/c/digit-recognizer)) in folder `experiments/MNIST`.
These can be used to test the GCP and GCP-based hyper-parameter optimization faster. See the `experiments` folder for more details on how to use the package for research purposes.  

### Understand Gaussian Copula Processes 
#### - `gcp_hpo.test`
In the `test` folder, you will find some utilities and scripts to either quantify the performances of GCP for regression purposes (eg. `test/regression/full_test.py`) or to display what it does.  

### What SmartSearch does  
![Fig1](fig/SmartSampling_example.png?raw=true)
*An example of the Smart Search process. The function to optimize is the blue line, and we start the process with 10 random points for which we know the real value (blue points). At each step, the performance function is modeled by a GCP and predictions are made (red crosses) based on the known data (blue and red points). The cyan zone shows the 95% condifence bounds. At each step the selected point (the one that maximizes the upper confidence bound) is shown in yellow. This point is then added to the known data so that the model becomes more and more accurate.*  


---------------

#### Contributor
[Sebastien Dubois](http://bit.do/sdubois)

#### Acknowledgments
* Many thanks to [Kalyan Veeramachaneni](http://www.kalyanv.org/) who originated this project during my visit at [Alfa Group](http://groups.csail.mit.edu/EVO-DesignOpt/groupWebSite/) (CSAIL, MIT), and for all his great advice.
* I would also like to thank Scikit-learn contributors as this code is based on Scikit-learn's GP implementation.
