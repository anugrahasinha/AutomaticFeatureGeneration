import os
import sys

import sqlalchemy.dialects.mysql.base as column_datatypes
from sqlalchemy.engine import create_engine
from sqlalchemy.schema import MetaData

import re
import ntiDSMGlobal as ntiGlb


class ntiDSMConfigGenerator:

	defaultConfig = {
				"entities" : {
				},
				"max_categorical_filter" : 2
			}

	categorical_limit_value = 20

	def __init__(self):
		self.databaseName = ""
		self.configFileName = ""
		try:
			(self.databaseName,self.configFileName) = self.getUserInput()
		except Exception as e:
			raise(Exception(str("ERROR : " + (str(e)))))
		self.finalConfig = self.defaultConfig
		self.db = None
		self.tableList=list()
		self.tableNameList=list()
		self.configFile = "config/"+self.configFileName
		self.engine = None
		self.check_metadata = None
		self.categorical_limit = self.categorical_limit_value
		ntiGlb.ntiDSMGlobalObj.log("Starting with config creation, database name given = %s" %(self.databaseName))
		try:
			if self.__initDBRead() == 0:
				self.__writeConfig()
		except Exception as e:
			raise(Exception(str("ERROR : " + (str(e)))))

	def getUserInput(self):
	        databaseName = raw_input("> Enter database name : ")
                if databaseName == None or databaseName == "":
			exceptionString = "> No Database given"
			print(exceptionString)
			ntiGlb.ntiDSMGlobalObj.log(exceptionString)
			raise(Exception(exceptionString))

	        configFile = raw_input("> Enter the config file name to written : config/")
                if configFile == None or configFile == "":
                        exceptionString = "> No Config given"
                        print(exceptionString)
                        ntiGlb.ntiDSMGlobalObj.log(exceptionString)
                        raise(Exception(exceptionString))

		if not re.search(".py",configFile):
			configFile = raw_input("> Config file needs to be have .py extension, try again : config/")
			if not re.search(".py", configFile):
				exceptionString = "Config file given does not have a .py extension"
				print(exceptionString)
				ntiGlb.ntiDSMGlobalObj.log(exceptionString)
				raise(Exception(exceptionString))
		return (databaseName,configFile)
	

	def __writeConfig(self):
		#print("config = " + ntiGlb.ntiDSMGlobalObj.printdict(self.finalConfig,stringAware=True))
		try:
			f = open(self.configFile,'w')
			f.write("config =" + ntiGlb.ntiDSMGlobalObj.printdict(self.finalConfig,stringAware=True))
			f.close()
		except Exception as e:
			raise(Exception("Unable to write final config file, exception : %s" %(str(e))))
		print("Config generated successfully.")
	def __initDBRead(self):
		url = 'mysql://necbuilder:necbuilder@123@10.0.1.26/%s' % (self.databaseName)
		try:
			self.engine = create_engine(url)
			self.check_metadata = MetaData(bind=self.engine)
			self.check_metadata.reflect()
		except Exception as e:
			exceptionString = "Problem connecting to database, exception : %s" %(str(e))
			ntiGlb.ntiDSMGlobalObj.log(exceptionString)
			raise(Exception(exceptionString))
		self.tableList = self.check_metadata.sorted_tables
		self.tableNameList = [x.name for x in self.check_metadata.sorted_tables]
		ntiGlb.ntiDSMGlobalObj.log("The tables present in this database are: %s" %(str(self.tableNameList)))
		if len(list(set(self.tableNameList))) != len(self.tableNameList):
			exceptionString = "There are duplicate table names in the database, please clear them. exiting...."
			ntiGlb.ntiDSMGlobalObj.log(exceptionString)
			print("> There are duplicate table names in the database, please clear them. exiting....")
			raise(Exception(exceptionString))
			
		## Iterate over table list to check if this is a database where we might have worked upon earlier ##
		print("> The tables present in this database are\n")
		for t1 in self.tableList:
			print(t1.name)
		try:
			print("\n> Checking if table list has some previously worked upon tables?\n")
			drop_tables = list(filter(lambda x : re.search("_ntidsm_[1-99999]",x), self.tableNameList))
			if len(drop_tables) > 0:
				ntiGlb.ntiDSMGlobalObj.log("Following tables found for removal : %s" %(str(drop_tables)))
				for x in drop_tables: print(x)	
				if self.__takeUserInputConfirm(message="Do you want to proceed with the above tables for removal: (Y/N) :") == False:
					ntiGlb.ntiDSMGlobalObj.log("User does not want to proceed, exiting....")
					print("> Exiting.....")
					exit(0)
				for t in drop_tables:
					try:
						## Dropping table in database ##
						qry = "drop table %s" %(t)
						self.engine.execute(qry)

						## determining the sql.alchemy table object for the table to be removed ##
						tableObj = list(filter(lambda x : x.name == t,self.tableList))
						## remove sql.alchemy table object from tableList##
						self.tableList.remove(tableObj[0])
						## removing tableName from tableNameList
						self.tableNameList.remove(t)
					except Exception as e:
						raise Exception("Exception in dropping table : %s" %(str(e)))
				print("\n> Table list after dropping tables\n")
				ntiGlb.ntiDSMGlobalObj.log("Table list after dropping tables : %s" %(str(self.tableNameList)))
				for tobj in self.tableList:
					print(tobj.name)
			else:
				print("> No tables found, which probably needs to be dropped")
		except Exception as e:
			exceptionString = "Exception is processing already present table, : %s \nexiting......" %(str(e))
			ntiGlb.ntiDSMGlobalObj.log(exceptionString)
			raise(Exception(exceptionString))

		if (os.path.isfile(self.configFile)):
			ntiGlb.ntiDSMGlobalObj.log("Config file : %s already exist, asking user for confirmation to update the file" %(self.configFile))
                        if not self.__takeUserInputConfirm("Do you want continue with config creation? (Y). To use old config (N) : (Y/N) :"):
                                return 1  ## Do not generate config further ##

		self.__initTablesRead()
		return 0 ## Generate config further ##
	def __initTablesRead(self):
		datatypes = [column_datatypes.INTEGER, column_datatypes.FLOAT, column_datatypes.DECIMAL, column_datatypes.DOUBLE, column_datatypes.SMALLINT, column_datatypes.MEDIUMINT]
		for t in self.tableList:
			self.finalConfig["entities"][t.name] = {
							"feature_metadata" : {
							},
							"included_row_functions" : [],
							"excluded_row_functions" : [],
							"excluded_agg_entities" : [],
							"one-to-one" : [],
						}
			num_rows_query = ("select count(*) from `%s`" %(t.name))
			num_rows = self.engine.execute(num_rows_query).fetchall()[0][0] 
			for col in t.c:
				self.finalConfig["entities"][t.name]["feature_metadata"][col.name] = {
												"categorical" : False,
												"numeric" : False,
												"categorical_filter" : False,
												"ignore" : False
											}
				self.finalConfig["entities"][t.name]["feature_metadata"][col.name]["numeric"] = type(col.type) in datatypes and not (col.primary_key or (len(col.foreign_keys)>0))
				if num_rows >= 10000000: ## ten million
					ntiGlb.ntiDSMGlobalObj.log("For table = %s, the number of rows = %d > 10million, no point looking at categorical" %(t.name,num_rows))
					continue
				## Building query to check categorical ##
				query = ("select count(distinct(`%s`)) from `%s`" %(col.name,t.name))
				urowcount = self.engine.execute(query).fetchall()[0][0]
				if urowcount <= self.categorical_limit:
					ntiGlb.ntiDSMGlobalObj.log("For table = %s, column = %s is a categorical filter, with unique row count = %d" %(t.name,col.name,urowcount))
					self.finalConfig["entities"][t.name]["feature_metadata"][col.name]["categorical"] = True
		ntiGlb.ntiDSMGlobalObj.log("Building config complete, for records, the config is")
		ntiGlb.ntiDSMGlobalObj.log(ntiGlb.ntiDSMGlobalObj.printdict(self.finalConfig,stringAware=True))
		

	def __takeUserInputConfirm(self,message=None):
		userInput = raw_input("> " + message)
		if userInput == "Y" or userInput == "y":
			return True
		elif userInput == "N" or userInput == "n":
			return False
		else:
			ntiGlb.ntiDSMGlobalObj.log("Unwanted input entered by user : %s.... Exiting.." %(str(userInput)))
			raise(Exception("Unwanted input entered"))
if __name__ == "__main__":

	ntiDSMConfigGenerator()



