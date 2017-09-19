### import ####
import os
import sys
import random
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import re

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import MultinomialNB
# from sklearn.feature_selection import SelectKBest, chi2

from gcp_hpo.smart_search import SmartSearch


### Set the parameters : ####
parameters = {'nb_features' : ['int', [1000, 40000]],
              # 'feature_select' : ['float', [0.5, 1.]],
              'max_ngram_range' : ['int', [1, 4]],
              'min_df' : ['float', [0., 0.3]],
              'max_df' : ['float', [0.4, 1.]],
              'alpha_NB' : ['float', [0.01, 1.]],
              'tfidf_norm' : ['cat', ['l1', 'l2']],
              'use_idf' : ['cat', ['T', 'F']]}


def review_to_wordlist(review):
    """
    Function to convert a document to a sequence of words.
    Returns a list of words.
    """
    # Remove HTML
    review_text = BeautifulSoup(review).get_text()
    #
    # Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    #
    # Convert words to lower case and split them
    words = review_text.lower().split()

    return(words)


def get_data():
    ### get data
    data = pd.read_csv(os.path.join('labeledTrainData.tsv'),
                       header = 0,
                       delimiter = "\t",
                       quoting = 3)

    clean_reviews = []
    for i in xrange( 0, len(data["review"])):
        clean_reviews.append(" ".join(review_to_wordlist(data["review"][i])))
    print "Data loaded"

    return data, clean_reviews


def scoring_function(parameters, data, Y, sub_size_):
    subsample_idx = range(25000)
    random.shuffle(subsample_idx)
    subsample_idx = subsample_idx[:(sub_size_)]

    subsample_clean_reviews = [data[i] for i in subsample_idx]
    sub_Y = np.asarray([Y[i] for i in subsample_idx])

    return scoring_function_cv(subsample_clean_reviews, sub_Y, parameters)


def scoring_function_cv(subsample_clean_reviews, Y, parameters):
    vectorizer = CountVectorizer(analyzer = "word",
                                 tokenizer = None,
                                 preprocessor = None,
                                 stop_words = None,
                                 max_features = parameters['nb_features'],
                                 ngram_range = (1, parameters['max_ngram_range']),
                                 max_df = parameters['max_df'],
                                 min_df =  parameters['min_df'])

    hist = vectorizer.fit_transform(subsample_clean_reviews)
    BOF_features = hist.toarray()
    
    tfidf = TfidfTransformer(norm = parameters['tfidf_norm'],
                             use_idf = (parameters['use_idf'] == 'T'),
                             smooth_idf = True,
                             sublinear_tf = False)
    X = tfidf.fit_transform(BOF_features)

    # final_nb_features = parameters['feature_select'] * X.shape[1]
    # print final_nb_features

    n_reviews = len(subsample_clean_reviews)

    ### CV ###
    kf = KFold(n_reviews,n_folds=5)
    cv_results = []
    for train_idx,test_idx in kf:
        X_cv,Y_cv = X[train_idx,:], Y[train_idx]
        
        # ch2 = SelectKBest(chi2, k = final_nb_features)
        # ch2.fit(X_cv, Y_cv)
        # chiSQ_val = ch2.scores_
        # index = np.argsort(chiSQ_val)[::-1]
        # best_feat = index[:final_nb_features]
        # X_cv = X_cv[:, best_feat]

        clf = MultinomialNB(alpha = parameters['alpha_NB'])
        clf.fit(X_cv, Y_cv)
        X_test = X[test_idx, :]
        # Y_pred = clf.predict_proba(X_test[:, best_feat])[:,1]
        Y_pred = clf.predict_proba(X_test)[:,1]
        res = roc_auc_score(Y[test_idx], Y_pred)
        cv_results.append(res)

    print 'CV res :', np.mean(cv_results)

    return cv_results


if __name__ == '__main__':
    n_test = int(sys.argv[1])
    sub_size = int(sys.argv[2])

    print 'Running', n_test, 'tests with a subsample size of', sub_size
    data, clean_reviews = get_data()

    def scoring(parameters):
        return scoring_function(parameters,
                                data = clean_reviews,
                                Y = data["sentiment"],
                                sub_size_ = sub_size)

    search = SmartSearch(estimator = scoring,
                         parameters = parameters,
                         model = 'rand',
                         n_init = n_test,
                         n_iter = n_test,
                         n_final_iter = 0,
                         detailed_res = 1)

    tested_params, outputs = search._fit()


    f = open(("scoring_function/" + str(sub_size) + "_output.csv"),'w')
    for line in outputs:
        print>>f,line

    np.savetxt("scoring_function/" + str(sub_size) + "_params.csv",
               tested_params,
               delimiter = ",")
