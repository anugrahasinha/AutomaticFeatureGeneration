import pdb
import profile

import os
import sys

from database import Database
import sqlalchemy.dialects.mysql.base as column_datatypes
import numpy as np
import agg_functions
import flat_functions
import row_functions
import datetime
import threading
import importlib
import re

## ASINHA IMPORTS ##
import ntiDSMGlobal as ntiGlb


#############################
# Table feature functions  #
#############################


"""
Make all feature of child tables

Make all agg features. Agg features are aggregate functions applied to child tables, so we must make those feature first, which we have donorschoose

Make row features. 

Make one to one features

Make a all features for parent tables. Parent tables use features of this table, so we must have calcualted agg, row, and one to one features.

Make flat features. Flat features pull from table features, so we must have made all feature for parent tables, which he have

"""


MAX_FUNC_TO_APPLY = 2

def make_all_features(db, table, caller=None, path=[], depth=0):
    print "TABLE: ", table, "\n"
    caller_name = 'no caller'
    if caller:
        caller_name = caller.name
        ntiGlb.ntiDSMGlobalObj.setCurrentTCD(table.name,caller.name,depth)
    else:
        ntiGlb.ntiDSMGlobalObj.setCurrentTCD(table.name,"MAIN",depth)
    threadName=ntiGlb.ntiDSMGlobalObj.getThreadName()
    print "*"*depth + 'making all features %s, path= %s \n' % (table.name, str(path))
    ntiGlb.ntiDSMGlobalObj.log('making all features %s, path= %s' % (table.name, str(path)))

    #found a cycle
    new_path = list(path) + [table]
    ntiGlb.ntiDSMGlobalObj.log("new_path value = " + str(new_path))
    if len(path) != len(set(path)):
        ntiGlb.ntiDSMGlobalObj.log("Found a cycle here, returning")
        return

    threads = []
    ntiGlb.ntiDSMGlobalObj.log("Current table = " + str(table.name) + " & output of table.get_child_tables " + str(table.get_child_tables()))
    for related,fk in table.get_child_tables():
        #dont make_all on the caller and dont make all on yourself
        if related != caller and related != table:
            ntiGlb.ntiDSMGlobalObj.log("Going in for recursive call on child table, parent table = %s and child table is %s" %(table.name,related.name))
            t = threading.Thread(target=make_all_features, args=(db, related), kwargs=dict(path=new_path, caller=table, depth=depth+1))
            # make_all_features(db, related, caller=table, depth=depth+1)
            t.start()
            t.join()
            if caller:
                ntiGlb.ntiDSMGlobalObj.setCurrentTCD(table.name,caller.name,depth)
            else:
                ntiGlb.ntiDSMGlobalObj.setCurrentTCD(table.name,"MAIN",depth)
            # threads.append(t)
        elif related == caller:
            ntiGlb.ntiDSMGlobalObj.log("Current Table = %s and related child table = %s and caller = %s, hence ignoring it" %(table.name,related.name,caller.name))
        else: ## related == table
            ntiGlb.ntiDSMGlobalObj.log("Current Table = %s and related child table = %s and caller = %s, cyclic dependency, hence ignoring it" %(table.name,related.name,caller.name))
    # [t.join() for t in threads]
    
    print "*"*depth +  'making agg features %s, caller= %s \n' % (table.name, caller_name)
    ntiGlb.ntiDSMGlobalObj.log('making agg features %s, caller= %s' % (table.name, caller_name))
    ntiGlb.ntiDSMGlobalObj.globalOperationInformation = ["Aggregate_Features",str(table.name),str(caller_name)]
    make_agg_features(db, table, caller, depth)
    ntiGlb.ntiDSMGlobalObj.log(threadName + " : " + "Making aggregate features finished")

    print "*"*depth +  'making row features %s \n' % (table.name)
    ntiGlb.ntiDSMGlobalObj.log('making row features %s' % (table.name))
    ntiGlb.ntiDSMGlobalObj.globalOperationInformation = ["Row_Features",str(table.name),str(caller_name)]
    make_row_features(db, table, caller, depth)
    ntiGlb.ntiDSMGlobalObj.log("Making row features finished")
    
    print "*"*depth +  'making one_to_one features %s \n' % (table.name)
    ntiGlb.ntiDSMGlobalObj.globalOperationInformation = ["One-on-One Features",str(table.name),str(caller_name)]
    ntiGlb.ntiDSMGlobalObj.log('making one_to_one features %s' % (table.name))
    make_one_to_one_features(db, table, caller, depth)
    ntiGlb.ntiDSMGlobalObj.log("Making one to one features finished")

    print "*"*depth +  'making flat features %s \n' % (table.name)
    ntiGlb.ntiDSMGlobalObj.log('making flat features %s' % (table.name))
    ntiGlb.ntiDSMGlobalObj.globalOperationInformation = ["Flat_Features_1st",str(table.name),str(caller_name)]
    make_flat_features(db, table, caller, depth) #Todo pass path so we don't bring in flat feature we do not need
    ntiGlb.ntiDSMGlobalObj.log("Making flat features finished")


    print "*"*depth + "Finished making features based on dependent table, now working with tables on which I depend on, current table name = %s" %(table.name)
    ntiGlb.ntiDSMGlobalObj.log("Finished making features based on dependent table, now working with tables on which I depend on, current table name = %s" %(table.name))

    ntiGlb.ntiDSMGlobalObj.log("Current table = " + str(table.name) + " & output of table.get_parent_tables " + str(table.get_parent_tables()))
    threads = []
    for related,fk in table.get_parent_tables():
        #dont make_all on the caller and dont make all on yourself
        if related != caller and related != table:
            t = threading.Thread(target=make_all_features, args=(db, related), kwargs=dict(path=new_path, caller=table, depth=depth+1))
            # make_all_features(db, related, caller=table, depth=depth+1)
            t.start()
            t.join()
            ### Resetting TCD as ntiDSMGlobalObj is a singleton object, and needs to be synced here ###
            if caller:
                ntiGlb.ntiDSMGlobalObj.setCurrentTCD(table.name,caller.name,depth)
            else:
                ntiGlb.ntiDSMGlobalObj.setCurrentTCD(table.name,"MAIN",depth)
            # threads.append(t)
        elif related == caller:
            ntiGlb.ntiDSMGlobalObj.log("Current Table = %s and related child table = %s and caller = %s, hence ignoring it" %(table.name,related.name,caller.name))
        else: ## related == table
            ntiGlb.ntiDSMGlobalObj.log("Current Table = %s and related child table = %s and caller = %s, cyclic dependency, hence ignoring it" %(table.name,related.name,caller.name))
    # [t.join() for t in threads]


    print "*"*depth +  'making flat features %s, caller= %s\n' % (table.name, caller_name)
    ntiGlb.ntiDSMGlobalObj.log('making flat features %s, caller= %s' % (table.name, caller_name))
    ntiGlb.ntiDSMGlobalObj.globalOperationInformation = ["Flat_Features_2nd",str(table.name),str(caller_name)]
    make_flat_features(db, table, caller, depth)
    ntiGlb.ntiDSMGlobalObj.log("Making flat features finished, after working on parent tables")
    print "make_all_features completed for: ", table, "\n\n"

#############################
# Agg feature      #
# Note from NTI : aggregate features will be made on the basis of following rules
#               : "table" -> In argument
#               : child_table -> tables which refers a column in "table" as a foriegn key
#               : 
# Condition     : 1. child_table mentioned in excluded_agg_entities in config skipped
#               : 2. If "table" has more than 10million rows or child_table has more than 10million rows, aggregation done (function is_one_to_one)
#               : 3. If number of distinct primarykeys of child_tables, which is linked to this table, has value 1 then it is one_to_one, skipp aggregate
#############################
def make_agg_features(db, table, caller, depth):

    ntiGlb.ntiDSMGlobalObj.log("in make_agg_features, ------ START ------")
    ## get_related_fks -> provides fkeys who fkey.parent.table.name will be the child table which refers to this "table" column as foreign key ##
    print "Get related fks for: ", table, " are : ", db.get_related_fks(table), '\n'
    ntiGlb.ntiDSMGlobalObj.log("in make_agg_features : Related foreign Keys (i.e column names of this table which are referred to as foriegn key by others) for table = %s are %s" %(str(table),str(db.get_related_fks(table))))
    print "excluded agg entities for : ", table, " are: ", table.config.get("excluded_agg_entities", []), '\n'
    ntiGlb.ntiDSMGlobalObj.log("in make_agg_features : excluded aggregate enteries as per config file are %s" %(str(table.config.get("excluded_agg_entities", []))))
    for fk in db.get_related_fks(table):
        
        print "executing make_agg_features with arguments ", table, " ", caller, " " , depth, '\n'
        ntiGlb.ntiDSMGlobalObj.log("in make_agg_features : table = %s caller = %s and depth = %s" %(str(table),str(caller),str(depth)))
        child_table = db.tables[fk.parent.table.name]
        print "Child table is ", child_table, '\n'
        ntiGlb.ntiDSMGlobalObj.log("in make_agg_features : foreign key = %s & Child table = %s" %(str(fk),child_table))

        if child_table.name in table.config.get("excluded_agg_entities", []):
            print "skip agg", child_table.name, table.name
            ntiGlb.ntiDSMGlobalObj.log("in make_agg_features : Skip aggregate as child table included in exclusion in config, config is %s" %(str(table.config.get("excluded_agg_entities", []))))
            continue

        print "Checking child_table ",child_table ," for one_to_one \n"       
        ntiGlb.ntiDSMGlobalObj.log("in make_agg_features : Checking child table = %s for one_to_one" %(str(child_table)))

        ## table.is_one_to_one(child,fk) : If "table" has more than 10million row or child has more than 10million row, consider aggregation
        ## Note NTI :
        ## is_one_to_one is based on the fact, that if self.table(parent) and related table(child) are JOINED and distinct number of child_table.primary_key
        ## and grouped by parent_table.primary_key is 1, that is for every primary key in Parent Table, there is 1 entry in Child table, then it is
        ## one_to_one_related

        if table.is_one_to_one(child_table, fk):
            ntiGlb.ntiDSMGlobalObj.log("in make_agg_features : Skipping Child Table %s as it is one_to_one" %(str(child_table)))
            continue

        print "Processing table " ,table, " and ", child_table, " as these are not one_to_one. \n"
        ntiGlb.ntiDSMGlobalObj.log("in make_agg_features : Processing table = %s and child_table = %s as they are not one_to_one" %(str(table),str(child_table)))
        # interval = table.config.get("make_intervals", {}).get(child_table.name, None)

        # if interval:
        #     agg_functions.make_intervals(db, fk, n_intervals=interval["n_intervals"], delta_days=interval["delta_days"])
        
        agg_functions.apply_funcs(db,fk)
    print "Execution of make_agg_features is complete. \n"
    ntiGlb.ntiDSMGlobalObj.log("in make_agg_features, ------ FINISHED ------")


#############################
# Flat feature    #
#############################
def make_flat_features(db, table, caller, depth):
    """
    add in columns from tables that this table has a foreign key to as well as make sure row features are made
    notes:
    - a table will only be flatten once
    - ignores flattening info from caller
    """
    ntiGlb.ntiDSMGlobalObj.log("in make_flat_features features, ------ START ------")
    ntiGlb.ntiDSMGlobalObj.log("in make_flat_features, with table name = %s and caller = %s" %(table.name,str(caller)))
    flat = flat_functions.FlatFeature(db)
    ntiGlb.ntiDSMGlobalObj.log("in make_flat_features, with table name = %s and caller = %s, list of foreign keys to iterate upon = %s" %(table.name,str(caller),str(table.base_table.foreign_keys)))
    for fk in table.base_table.foreign_keys:
        parent_table = db.tables[fk.column.table.name]
        if parent_table in [table, caller]:
            continue
        
        flat.apply(fk)
    ntiGlb.ntiDSMGlobalObj.log("in make_flat_features features, ------ FINISHED ------")

#############################
# Flat feature    #
#############################
def make_one_to_one_features(db, table, caller, depth):
    ntiGlb.ntiDSMGlobalObj.log("in make_one_to_one features, ------ START ------")
    print "Calling make_one_to_one_features for ", table , " ",caller, '\n'
    ntiGlb.ntiDSMGlobalObj.log("in make_one_to_one features, with table name = %s caller = %s" %(table.name,str(caller)))
    flat = flat_functions.FlatFeature(db)
    ntiGlb.ntiDSMGlobalObj.log("in make_one_to_one features, for table name = %s, child table & foreign key tuples = %s" %(table.name,str(table.get_child_tables())))
    for related, fk in table.get_child_tables():
        if table.is_one_to_one(related, fk):
            ntiGlb.ntiDSMGlobalObj.log("in make_one_to_one features, for table name = %s, related table = %s is one_to_one" %(table.name,related.name))
            one_to_one_table = db.tables[fk.parent.table.name]
            ntiGlb.ntiDSMGlobalObj.log("in make_one_to_one features, finally, for table name = %s, one_to_one table = %s" %(table.name,one_to_one_table.name))
            if one_to_one_table in [table, caller]:
                print "Cannot execute one_to_one for ", table, " " , caller, '\n'
                ntiGlb.ntiDSMGlobalObj.log("in make_one_to_one features, one_to_one_table = %s, is the caller or current table name, current table = %s & caller = %s SKIPPING IT....." %(one_to_one_table.name,table.name,str(caller)))
                continue
            print "Executing one_to_one for ", table, " " , caller, '\n'
            ntiGlb.ntiDSMGlobalObj.log("in make_one_to_one feature, table = %s, caller = %s & one_to_one_table = %s, now applying one_to_one functions" %(table.name,str(caller),one_to_one_table.name))
            flat.apply(fk, inverse=True)
    ntiGlb.ntiDSMGlobalObj.log("in make_one_to_one features, ------ FINISHED ------")
        

#############################
# Row feature functions     #
#############################
def make_row_features(db ,table, caller, depth):
    # pass
    ntiGlb.ntiDSMGlobalObj.log("in make_row_features, ------ START ------")
    ntiGlb.ntiDSMGlobalObj.log("in make_row_features, with table name = %s" %(table.name))
    row_functions.apply_funcs(table)
    ntiGlb.ntiDSMGlobalObj.log("in make_row_features, ------ FINISHED ------")
    # add_ntiles(table)

def ntiDSMStartFeatureGenEngine(database_name=None,user_input_config=None):
    if database_name == None or database_name == "" or user_input_config == "" or user_input_config == None:
        exceptionString="Database or config not provided"
        #ntiGlb.ntiDSMGlobalObj.log(exceptionString)
        raise(Exception(exceptionString))

    import debug
    from sqlalchemy.engine import create_engine
    from sqlalchemy.schema import MetaData

    #database_name = 'kdd2015'

    ### NTI : Take user inputs ####

    #user_input_db=raw_input("Enter the database name (default : kdd2015) :")
    #if (user_input_db != "" ):
    #    database_name = user_input_db

    table_name = "Enrollments"
    user_input_table = raw_input("Enter starting table name (default : Enrollments) : ")
    if (user_input_table != "" ):
        table_name = user_input_table
    
    config_file = ["blank_config.py"]
    if (user_input_config != ""):
        config_file[0] = user_input_config


    ntiGlb.ntiDSMGlobalObj.log("User entered database name %s and table name %s and config file %s" %(database_name,table_name,config_file))
    save_name = "models/"+database_name + "__" + table_name
    url = 'mysql://necbuilder:necbuilder@123@10.0.1.26/%s' % (database_name)
    engine = create_engine(url)
    drop_tables = ["courses_1", "categoryTypes_1", "Enrollments_1", "sourceTypes_1", "Log_1", "eventTypes_1", "objectChildren_1", "objects_1", "users_1", "outcomes_1"]
    
    print("\nTrying to import the config file\n")

    config=dict()

    try:
        user_module_import = list(map(importlib.import_module,["config."+((c.split("."))[0]) for c in config_file]))
        print("Imported modules are : %s" %(str(user_module_import)))
        config = user_module_import[0].config
    except Exception as e:
        print("FATAL error:  \"%s\" : in importing modules, exiting(255)" %(str(e)))
        sys.exit(255)
        
    #from kdd2015_config import config
    #from superstoresdb_test1 import config
    yes = raw_input("Continue %s (y/n): " % database_name)
    if yes == "y":
        #original_db = Database('mysql://necbuilder:necbuilder@123@10.0.1.26/%s' % (database_name), config=config)
        check_metadata = MetaData(bind=engine)
        check_metadata.reflect()
        check_all_tablenames = [x.name for x in check_metadata.sorted_tables]
        if table_name in check_all_tablenames:
            drop_tables = list(filter(lambda x : re.search("_ntidsm_[1-99999]",x) != None ,check_all_tablenames))
            if len(drop_tables) > 0:
                print("Seems I have worked on this database previously, dropping old analyzed tables with \"_1\" as the table name")
                print("Drop table list = %s" %(str(drop_tables)))
                confirm = raw_input("Drop the mentioned tables (y/n): ")
                if confirm == "y" or confirm == "Y":
                    for t in drop_tables:
                        try:
                            qry = "drop table %s" % t
                            engine.execute(qry)
                        except Exception, e:
                            print e
                else:
                    print("Since old data is still present, not continuing. exit(255)")
                    exit(255)
        else:
            print("Table name entered %s is not present in database.\n Available tables in database are : %s" %(table_name,list(filter(lambda x : x.find("_1") == -1 ,check_all_tablenames))))
            exit(255)
         
        #reloaded db after dropping tables
        db = Database('mysql://necbuilder:necbuilder@123@10.0.1.26/%s' % (database_name), config=config) 
        table = db.tables[table_name]
        ## FINAL TRIGGER ##
        make_all_features(db, table)
        ## ------------- ##

	ntiGlb.ntiDSMGlobalObj.log("Final global column modification dict data = \n %s" %(str(ntiGlb.ntiDSMGlobalObj.globalprint(ntiGlb.ntiDSMGlobalObj.globalColumnModificationDict))))
	tableColumnDropDict = ntiGlb.ntiDSMGlobalObj.dropColumns()
	#ntiGlb.ntiDSMGlobalObj.log("Drop Column Dictionary is = \n %s" %(str(tableColumnDropDict)))
	for table in tableColumnDropDict.keys():
		## tableColumnDropDict has <TableName> : [list of columns to be removed]
		## We iterate of over the [list of columns to be removed] through a lambda function
		## and build a DROP query information. This will be in the form of a list as ["DROP users_1__799,", "DROP users_1__798,"...]
		## We then perform a join of the list of above list using spaces
		
		finalDropString = " ".join(list(map(lambda x : "" + ("DROP %s__%d," %(table,x)),tableColumnDropDict[table])))
		
		## Since the last character of the joined string will have ",", removing it
		finalDropString = finalDropString[0:len(finalDropString)-1]

		## Building the final query
		query = "ALTER TABLE %s %s" %(table,finalDropString)
		#ntiGlb.ntiDSMGlobalObj.log("Dropping from table = %s with query = \n%s" %(table,query))
		try:
			engine.execute(query)
		except Exception as e:
			ntiGlb.ntiDSMGlobalObj.log("Exception in removing from table = %s and exception = %s" %(table,str(e)))

	ntiGlb.ntiDSMGlobalObj.log("Final columns of table = %s_1 to refer and there corresponding functions" %(table_name))
	for baseTable in ntiGlb.ntiDSMGlobalObj.globalColumnModificationDict.keys():
		keysToConsider = list(filter(lambda x : x[0] == str(table_name + "_1"),ntiGlb.ntiDSMGlobalObj.globalColumnModificationDict[baseTable].keys()))
		for key in keysToConsider:
			ntiGlb.ntiDSMGlobalObj.log("Table = %s, column = %s : func = %s" %(key[0],key[1],ntiGlb.ntiDSMGlobalObj.globalColumnModificationDict[baseTable][key]))

	ntiGlb.ntiDSMGlobalObj.log("Final Statistics : %s" %(str(ntiGlb.ntiDSMGlobalObj.printdict(ntiGlb.ntiDSMGlobalObj.globalFeatureGenerationStatistics))))

        #exit(0) 
        db.save(save_name)
        print("Features generated!\n")

    
        db = Database.load(save_name)
        table = db.tables[table_name]

        debug.export_col_names(table)
        print("Execution completed!\n")

    # debug.print_cols_names(db.tables['Orders'])

