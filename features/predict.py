from make_features import make_all_features
from filters import FilterObject

from sklearn.utils import ConvergenceWarning
import warnings
from database import Database
import numpy as np
from sklearn import tree
from sklearn import cross_validation
from sklearn import preprocessing
import random
from sklearn.cluster import MiniBatchKMeans
from time import time
from sklearn.preprocessing import scale
from sklearn.manifold import TSNE
from sklearn.cross_validation import train_test_split
from scipy import sparse
from itertools import izip
import threading
import pdb
from collections import OrderedDict
import cPickle as pickle
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from DeepMining.smart_sampling import smartSampling

from learning import DSMClassifier

import ntiDSMGlobal as ntiGlb

def get_predictable_features(table):
    def match_func(col):
        exclude = table.config.get("excluded_predict_entities", [])
        if not (col.metadata['numeric'] or col.metadata['categorical']):
            ntiGlb.ntiDSMGlobalObj.log("Column : %s dropped -> numeric = %s and categorical = %s" %(col.metadata['real_name'],str(col.metadata['numeric']),str(col.metadata['categorical'])))
            return False
        
        table_path = set([p['base_column'].dsm_table.name for p in col.metadata["path"]])
        if col.metadata["path"] and table_path.intersection(exclude) != set([]):
            ntiGlb.ntiDSMGlobalObj.log("Column : %s dropped as it is part of excluded_predict_entities = %s" %(col.metadata['real_name'],str(exclude)))
            return False


        #hack for fixing label in ijcai
        for p in col.metadata["path"]:
            if p["base_column"].dsm_table.name == "Users":
                for p2 in p["base_column"].metadata["path"]:
                    if p2.get('filter',None):
                        if p2["filter"].filtered_cols[0].name == "label":
                            return False

        return True

    return table.get_column_info(match_func=match_func)

def get_usable_features(target_col):
    def match_func(col, target_col=target_col):
        target_col_path = set([p['base_column'].unique_name for p in target_col.metadata['path']])
        usable_col_path = set([p['base_column'].unique_name for p in col.metadata['path']])
        
        if target_col_path.intersection(usable_col_path) != set([]):
            return False

        if col.metadata.get("interval_num", None) != None:
            return False

        if not (col.metadata['numeric'] or col.metadata['categorical']):
            return False

        if col == target_col:
            return False

        if len(col.get_distinct_vals()) < 2:
            return False

        return True

    return target_col.dsm_table.get_column_info(match_func=match_func)

def make_data(target_col, feature_cols, id_col, filter_obj=None, inc_unlabeled=True, cache=False, limit=None):
    """
    todo: make this one query
    """
    ntiGlb.ntiDSMGlobalObj.log("Argument information :\n\
    target_col=%s\n\
    len(feature_cols)=%s\n\
    id_col=%s\n\
    filter_obj=%s\n\
    inc_unlabeled=%s\n\
    cache=%s\n\
    limit=%s\n"\
    %(target_col.name,len(feature_cols),id_col.name,str(filter_obj),str(inc_unlabeled),str(cache),str(limit)))
    CACHE_FILE = "./models/cache/make_data_cache.p"
    if cache:
        #print "using cache"
        ntiGlb.ntiDSMGlobalObj.log("Cache option is TRUE, using cache")
        cache_data = pickle.load( open(CACHE_FILE, "rb" ) )
        return cache_data[0], cache_data[1], cache_data[2], cache_data[3]

    table = target_col.dsm_table
    ntiGlb.ntiDSMGlobalObj.log("For target_col = %s, the table chosen with target_col.dsm_table = %s" %(target_col.name,target_col.dsm_table.name))
    if not inc_unlabeled:
        f = FilterObject([(target_col, " IS NOT NULL", None )])
        if filter_obj != None:
            filter_obj = filter_obj.AND(f)
        else:
            filter_obj = f

    # print filter_obj.to_where_statement()

    id_col_vals = []
    y = []
    rows = []
    row_data = table.get_rows([target_col, id_col]+feature_cols, filter_obj=filter_obj, limit=limit)

    ## NTI Note : row_data is not a list, it is a <class 'sqlalchemy.engine.result.ResultProxy'>"
    ##          : Therefore, we cannot check the row count by using len(row_data). Sqlalchemy has a function
    ##          : defined in /usr/lib64/python2.7/site-packages/sqlalchemy/engine/result.py, def rowcount
    ##          : Using this function to check the total rows fetched
    ntiGlb.ntiDSMGlobalObj.log("Data fetched from table = %s has rows = %d" %(table.name,row_data.rowcount))
    for r in row_data:
        y_add = r[0]
        id_add = r[1]
        # print r
        if y_add == None and inc_unlabeled:
            y_add = 0 
        elif y_add == None:
            continue

        #print("ANUGRAHA 5: type(y_add) : %s, type(id_add) : %s" %(str(type(y_add)),str(type(id_add))))
        y.append(float(y_add)) 
        id_col_vals.append(id_add) 
        rows.append(r[2:])
    rows = np.array(rows)

    ntiGlb.ntiDSMGlobalObj.log("rows.shape = %s, len(y) = %d, len(id_col_vals) = %d" %(str(rows.shape),len(y),len(id_col_vals)))

    cat_mask = [col.metadata["categorical"] for col in feature_cols]
    le = preprocessing.LabelEncoder()
    for i,is_cat in enumerate(cat_mask):
        if is_cat:
            ntiGlb.ntiDSMGlobalObj.log("Colname = %d is a categorical column, perform sklearn fit_transform" %(feature_col[i].name))
            rows[:,i] = le.fit_transform(rows[:,i])
    rows = np.array(rows, dtype=np.float)
    cat_mask = np.array(cat_mask, dtype=bool)
    y = np.array(y, dtype=int)

    if cache:
        cache_data = [rows,y, id_col_vals, cat_mask]
        pickle.dump(cache_data, open( CACHE_FILE, "wb" ))

    ntiGlb.ntiDSMGlobalObj.log("Final output from make_data, rows.shape = %s, y.shape = %s, len(id_col_vals) = %s, len(cat_mask) = %d" %(str(rows.shape),str(y.shape),len(id_col_vals),len(cat_mask)))
    return rows,y, id_col_vals, cat_mask
    # num = 10000
    # return rows[:num], y[:num]

def split_data(x,y, ids, test_ids_file, id_type=int):
    """
    splits training/test data according to the list of ids in test_ids_file
    """
    with open(test_ids_file) as f:
        test_id_set = set(map(id_type,f.read().splitlines()))

    y = np.array(y)
    ids = np.array(ids)

    test_idx = []
    train_idx = []
    for idx, curr_id in enumerate(ids):
        if curr_id in test_id_set:
            test_idx.append(idx)
        else:
            train_idx.append(idx)
    train_x = x[train_idx]
    test_x = x[test_idx]
    train_y = y[train_idx]
    test_y = y[test_idx]
    train_ids = ids[train_idx]
    test_ids = ids[test_idx]

    ntiGlb.ntiDSMGlobalObj.log("train_x.shape = %s, test_x.shape = %s, train_y.shape = %s, test_y.shape = %s, train_ids.shape = %s, test_ids.shape = %s" %(str(train_x.shape),str(test_x.shape),str(train_y.shape),str(test_y.shape),str(train_ids.shape),str(test_ids.shape)))

    # pdb.set_trace()
    return train_x, test_x, train_y, test_y, train_ids, test_ids

def cluster_labels(X, k, scale_data=True):
    X = np.where(X == np.array(None), 0, X)

    if scale_data:
        X = scale(X)

    mbk = MiniBatchKMeans(init='k-means++', n_clusters=k, batch_size=1000,
                      n_init=10, max_no_improvement=10, verbose=1,
                      random_state=0)
    labels = mbk.fit_predict(X)
    return labels, X

def cluster(entity,k,cols=None):
    if cols == None:

        cols = entity.get_column_info(match_func=lambda col: col.metadata['numeric'])
    cols = np.array(cols)

    data = entity.get_rows(cols, limit=1000)
    X = [d for d in data]
    X = np.where(X == np.array(None), 0, X)
    labels, X_scaled = cluster_labels(X, k)
    # pdb.set_trace()
    
    important_features = {}
    distinct_labels = set(labels)
    for l in distinct_labels:
        predict_labels = labels.copy()
        predict_labels[labels == l] = 1
        predict_labels[labels != l] = 0
        
        clf = LinearSVR()
        clf.fit(X_scaled, predict_labels)
        score = clf.score(X_scaled, predict_labels)
        weights = clf.coef_
        weights = [abs(w) for w in weights]
        thresh = np.mean(weights) + np.std(weights)*3

        important_features[str(l)] = []
        for col, weight in zip(cols, weights):
            if weight > thresh:
                important_features[str(l)].append([col,weight])
    
        important_features[str(l)].sort(key=lambda x: -abs(x[1]))

    # reduced_data = PCA(n_components=2).fit_transform(X_scaled)
    reduced_data = PCA(n_components=50).fit_transform(X_scaled)
    tsne = TSNE(n_components=2, random_state=0,verbose=1)
    reduced_data = tsne.fit_transform(reduced_data) 
    # kmeans = MiniBatchKMeans(init='k-means++', n_clusters=k, n_init=10)
    # labels = kmeans.fit_predict(reduced_data)

    clusters = {}
    for pt,label in zip(reduced_data, labels):
        label = str(label)
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(list(pt))

    return clusters, important_features

def write_predictions(test_x, test_ids, clf, outfile, header=None, convert_id_func=None, name=""):
    all_probs = clf.predict_proba(test_x)
    # clf = self.cluster_clf[distinct_label].named_steps["clf"]
    # if clf.classes_[0] == 1:
    #     pos_idx = 0
    # elif clf.classes_[1] == 1:
    #     pos_idx = 1
    all_probs = all_probs[:,1]
    outfile2 = outfile + name + ".csv"
    with open(outfile2, "w") as out:
        if header:
            out.write(header+ "\n")
        if convert_id_func != None:
            test_ids = convert_id_func(test_ids)

        for row_id, prob in izip(test_ids, all_probs):
            out.write(str(row_id) + "," + str(prob) + "\n")

def do_trial(target_feature, predictors, params={}, filter_obj=None, cv=3, limit=None):
    predictors = list(predictors)
    id_col = target_feature.dsm_table.get_primary_key()
    X, y, ids, cat_mask = make_data(target_feature, predictors, id_col, filter_obj, inc_unlabeled=False, limit=limit)
    print  X, y
    clf = DSMClassifier(cat_mask, params)
    scores = cross_validation.cross_val_score(clf, X, y=y, scoring="roc_auc", cv=cv, n_jobs=-1, verbose=1, pre_dispatch=3)
    # def convert_id_func(row_ids):
    #     with open("../datasets/ijcai/test_ids_convert.csv") as f:
    #         return np.array(f.read().splitlines())
    # model(id_col,target_feature,predictors, params=params, outfile="ijcai_predict", header="user_id,merchant_id,prob", convert_id_func=convert_id_func, test_ids_file="../datasets/ijcai/test_ids.csv",id_type=int)
    # model(id_col,target_feature,predictors, filter_obj=filter_obj, params=params, outfile="donors_predict", header="projectid,is_exciting", test_ids_file="../datasets/donorschoose/test_ids.csv", id_type=str)
    model(id_col,target_feature,predictors, params=params, outfile=output_file, test_ids_file=file_test_ids, id_type=int)
    return scores.mean()

def choose_model_params(X, y, cat_mask, param_ranges):
    #0 Best parameters [ 4 10 38  8 79  8] with output: 0.630550591104

    def scoring_function(params_array):
        params_array = map(int, params_array)
        test_params = OrderedDict(zip(param_names,params_array))
        clf = DSMClassifier(cat_mask, test_params)
        print test_params
        scores = cross_validation.cross_val_score(clf, X, y=y, scoring="roc_auc", cv=3, n_jobs=1, verbose=1, pre_dispatch=3)
        print "*********"
        print test_params
        print scores
        print scores.mean()
        print "*********"  

        return scores.mean()

    param_names = param_ranges.keys()
    params = np.array(param_ranges.values())
    best_params = smartSampling(score_function=scoring_function, parameter_bounds = params, nb_random_steps=30, nb_iter=50, model="GCP", verbose=1)
    best_params = best_params.tolist()[0]
    best_params = OrderedDict(zip(param_names,best_params))
    return  best_params
        
def model(id_col, target_col, feature_cols, params={}, choose_params = None, filter_obj=None, outfile=None, header=None, convert_id_func=None, test_ids_file=None, id_type=int):
    ntiGlb.ntiDSMGlobalObj.log("-------- Starting with model function ------")
    pdb.set_trace()
    ntiGlb.ntiDSMGlobalObj.log("Argument information : \n\
    id_col = %s\n\
    target_col = %s\n\
    len(feature_cols) = %d\n\
    param = %s\n\
    choose_params=%s\n\
    filter_obj=%s\n\
    outfile=%s\n\
    header=%s\n\
    convert_id_func=%s\n\
    test_id_file=%s\n\
    id_type=%s"\
    %(id_col.name,target_col.name,len(feature_cols),ntiGlb.ntiDSMGlobalObj.printdict(params),str(choose_params),str(filter_obj),str(outfile),str(header),str(convert_id_func),str(test_ids_file),str(id_type)))
    num_features = len(feature_cols)
    feature_cols = list(feature_cols)

    ntiGlb.ntiDSMGlobalObj.log("Length of feature columns = %d" %(len(feature_cols)))

    #print "make data start"
    # print filter_obj
    ntiGlb.ntiDSMGlobalObj.log("-------- Starting with make_data function ------")
    rows, y, ids, cat_mask = make_data(target_col, feature_cols, id_col, filter_obj)
    ntiGlb.ntiDSMGlobalObj.log("-------- End make_data function ------")
    #print "make data done"
    # rows = np.nan_to_num(rows)
    print "split start"
    ntiGlb.ntiDSMGlobalObj.log("-------- Starting with split data function ------")
    train_x, test_x, train_y, test_y, train_ids, test_ids = split_data(rows,y, ids, test_ids_file=test_ids_file, id_type=id_type)
    ntiGlb.ntiDSMGlobalObj.log("-------- End split data function ------")
    print "split done"
    # print "\n".join([str(x) for x in zip(feature_cols,test_x[4])])
    # pdb.set_trace()    
    #whether or not to split test set to generate cross validation score
    if choose_params != None:
        params = choose_model_params(train_x, train_y, cat_mask, choose_params)

    ntiGlb.ntiDSMGlobalObj.log("------- Building NTI-DSM Classifier Obj, based on cat_mask and params set --------")
    clf = DSMClassifier(cat_mask, params)
    ntiGlb.ntiDSMGlobalObj.log("------- NTI-DSM Classifier Obj creation done. --------")
    clf.fit(train_x, train_y)          
    params = clf.params
    if outfile:
        name = zip(params.keys(), params.values())
        name = [str(n[0]) + "_" + str(n[1]) for n in name]
        name = ",".join(name)
        write_predictions(test_x, test_ids, clf, outfile, header=header, convert_id_func=convert_id_func, name=name)

    return params


if __name__ == "__main__":
    import os

    ntiGlb.ntiDSMGlobalObj.log("------------ Starting with prediction -------------")
    params = OrderedDict()
    params["k"] = [1,6]
    params["n_components"] = [10,500] #todo make sure we always have at least one feature
    params["k_best_percent"] = [10,100]
    # params["gbm_depth"] = [1,10]
    # params["gbm_subsample"] = [1,100]
    params["reweight_ratio"] = [1,10]
    params["n_estimators"] = [50, 500]
    params["max_depth"] = [1, 20]
    params["max_features"] = [1, 100]

    # ('k', 4), ('k_best', 91), ('n_components_percent', 89), ('reweight_ratio', 4), ('n_estimators', 432), ('max_depth', 7)])
    # [  3  46  93   1 327   9]
    # Best parameters [  2  27  87   1 228  10  57] with output: 0.672960205525
    # params = OrderedDict()
    # params["k"] = 2
    # params["k_best"] = 27 
    # params["n_components_percent"] = 87
    # params["reweight_ratio"] = 2
    # params["n_estimators"] = 338
    # params["max_depth"] = 10
    # params["max_features"] = 57

    #OrderedDict([('k', 1), ('n_components', 271), ('k_best_percent', 65), ('reweight_ratio', 8), ('n_estimators', 453), ('max_depth', 10), ('max_features', 51)])
    #jcai best 0 Best parameters [  1 271  65   8 453  10  51] with output: 0.654014087415
    #TEMP KDD2015 BEST  [  1 475  46   2 327  14  11]
    #best kdd2015 0 Best parameters [  3 384  18   1 381   8  76] with output: 0.866633804505
    #0               Test paramter: [  1 496  25   1 369  12  14]  - ***** accuracy: 0.866605058394

    # params = OrderedDict()
    params["k"] = 1
    params["n_components"] = 496
    params["k_best_percent"] = 25
    params["reweight_ratio"] = 1
    params["n_estimators"] = 369
    params["max_depth"] = 12
    params["max_features"] = 14



    # table_name = "Outcomes"
    # model_name = "ijcai__" + table_name
    # print "models/%s"%model_name
    # db = Database.load("models/%s"%model_name)
    # print "db loaded"
    # pdb.set_trace()
    # table = db.tables[table_name]
    # target_col = table.get_col_by_name("label")
    # id_col = table.get_col_by_name("id")
    # feature_cols = get_predictable_features(table)

    # feature_cols = [ c for c in feature_cols if c!= target_col]
    # def convert_id_func(row_ids):
    #     with open("../datasets/ijcai/test_ids_convert.csv") as f:
    #         return np.array(f.read().splitlines())
    
    # # params = model(id_col,target_col,feature_cols, choose_params=params, outfile="ijcai_predict", header="user_id,merchant_id,prob", convert_id_func=convert_id_func, test_ids_file="../datasets/ijcai/test_ids.csv",id_type=int)
    # # params = {}
    # params = model(id_col,target_col,feature_cols, params=params, outfile="ijcai_predict", header="user_id,merchant_id,prob", convert_id_func=convert_id_func, test_ids_file="../datasets/ijcai/test_ids.csv",id_type=int)
    # score = do_trial(target_col,feature_cols,params=params)
    # print score, params

    # table_name = "Projects"
    # model_name = "donorschoose__" + table_name
    # print "models/%s"%model_name
    # db = Database.load("models/%s"%model_name)
    # print "db loaded"
    # table = db.tables[table_name]
    # target_col = table.get_col_by_name("Outcome.is_exciting")
    # id_col = table.get_col_by_name("projectid")
    # feature_cols = get_predictable_features(table)
    # feature_cols = [ c for c in feature_cols if c!= target_col]
    # filter_col = table.get_col_by_name("date_posted")
    # filter_obj = FilterObject([(filter_col, ">", "2013-1-1")])
    # # filter_obj=None
    # # params = model(id_col,target_col,feature_cols, filter_obj=filter_obj, choose_params=params, outfile="donors_predict", header="projectid,is_exciting", test_ids_file="../datasets/donorschoose/test_ids.csv", id_type=str)
    # params = model(id_col,target_col,feature_cols, filter_obj=filter_obj, params=params, outfile="donors_predict", header="projectid,is_exciting", test_ids_file="../datasets/donorschoose/test_ids.csv", id_type=str)
    # # print "do trial"
    # print do_trial(target_col,feature_cols, filter_obj=filter_obj,params=params)
    # print params

    ### NTI DSM ###
    ### Changes for taking user input ####
    
    table_name = raw_input("Please enter the Primary Table name for analysis : ")
    if (table_name == ""):
        print("Table name is empty. Exiting....")
        exit(1)

    model_name = raw_input("Please enter the model name generated by NTI DSM Framework : models/")
    if (model_name == ""):
        print("Model name is empty. Exiting....")
        exit(1)

    target_col_name = raw_input("Please enter the flattened Outcomes column name, such as <tablename>.<colname> : ")
    if (target_col_name == ""):
        print("Target column name is empty.Exiting....")
        exit(1)

    id_col_name = raw_input("Please enter the primary identification column name in table = %s : " %(table_name))
    if (id_col_name == ""):
        print("Primary id column name is empty.Exiting....")
        exit(1)
    
    global file_test_ids
    file_test_ids = raw_input("Please enter the test ids file, i.e. CSV file where\nprimary identification ids are mentioned for which prediction is required (Complete Path, please):")
    if file_test_ids == "":
        print("Test IDs file given is empty. Exiting....")
        exit(1)

    global output_file
    output_file = raw_input("Please enter output file name, where you want prediction output (Complete path, please):")
    if output_file == "":
        print("Output file is empty. Exiting....")
        exit(1)

    #table_name = "Enrollments"
    #model_name = "kdd2015__" + table_name
    #model_name = "kdd2015_asinha_test_train1__Enrollments"
    #model_name = "kdd2015_akaur_test_train1__Enrollments"
    ntiGlb.ntiDSMGlobalObj.log("Model being used : models/%s & table name = %s" %(model_name,table_name))
    print "models/%s"%model_name
    db = Database.load("models/%s"%model_name)
    print "db loaded"
    table = db.tables[table_name]


    #target_col_name="Outcome.dropped_out"
    ntiGlb.ntiDSMGlobalObj.log("target column name : %s" %(target_col_name))

    target_col = table.get_col_by_name(target_col_name)


    #id_col_name = "enrollment_id"
    ntiGlb.ntiDSMGlobalObj.log("id column name : %s" %(id_col_name))

    id_col = table.get_col_by_name(id_col_name)
    feature_cols = get_predictable_features(table)
    feature_cols = [ c for c in feature_cols if c!= target_col]
    ntiGlb.ntiDSMGlobalObj.log("Length feature_cols = %d" %(len(feature_cols)))

    # params = model(id_col,target_col,feature_cols, choose_params=params, outfile="kdd2015_predict", test_ids_file="../datasets/kdd2015/test_ids.csv", id_type=int)

    ntiGlb.ntiDSMGlobalObj.log("----- Starting param generation using model function -----")
    ntiGlb.ntiDSMGlobalObj.log("Default value param being sent is \n %s" %(ntiGlb.ntiDSMGlobalObj.printdict(params)))

    params = model(id_col,target_col,feature_cols, params=params, outfile=output_file, test_ids_file=file_test_ids, id_type=int)
    #### NTI DSM : TEMPORARY TERMINATION for ANALYSIS ####
    #exit(1)
    print "do trial"
    # for f in feature_cols:
    #     print f
    print do_trial(target_col,feature_cols,params=params)
    print params
