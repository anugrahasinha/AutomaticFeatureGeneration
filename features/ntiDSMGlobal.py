import os
from datetime import datetime

class ntiDSMGlobal:
	def __init__(self):
		self.globalColumnModificationDict=dict()
		self.globalFeatureGenerationStatistics=dict()
		self.globalCurrentTableCallerDepth = ["MAIN","BASE",0]
		self.globalOperationInformation = list()
		self.currentThreadName="THREAD_BASE_MAIN"
		if not os.path.isdir("./ntilog"):
			os.makedirs("./ntilog")
			f = open("./ntilog/iterator",'w')
			f.write("0\n")
			f.close()
		elif not os.path.isfile("./ntilog/iterator"):
			f.open("./ntilog/iterator",'w')
			f.write("0\n")
			f.close()
		logIteratorFileObj = open("ntilog/iterator")
		newIteratorValue = int(logIteratorFileObj.read().strip()) + 1
		logIteratorFileObj.close()
		logIteratorFileObj = open("ntilog/iterator",'w')
		logIteratorFileObj.write(str(newIteratorValue))
		logIteratorFileObj.close()
		self.logfile = "./ntilog/nti_"+str(newIteratorValue)+".log"
		print("NTI LOG : Refer log file = "+self.logfile)
		self.loglevel = "INFO"


	def setCurrentTCD(self,table,caller=None,depth=1):
		self.globalCurrentTableCallerDepth = [table,caller,depth]
		if caller:
			self.currentThreadName = "THREAD_" + caller + "_" + table
		else:
			self.currentThreadName = "THREAD_MAIN_" + table

	def getThreadName(self):
		return self.currentThreadName

	def getCurrentTCD(self,info=0):
		return self.globalCurrentTableCallerDepth[info]

        def log(self,message=None,loglevel="INFO"):
		if (loglevel == "INFO" or loglevel == self.loglevel):
	                dt = datetime.now()
	                if message != None:
	                        logString='{0:^20}'.format(dt.strftime("%Y-%m-%d_%H:%M:%S")) + " : " + self.currentThreadName + " : Depth : " + "*"*self.getCurrentTCD(2) + " : " + message + "\n"
	                else:
	                        logString='{0:^20}'.format(dt.strftime("%Y-%m-%d_%H:%M:%S")) + " : " + self.currentThreadName + " : Depth : " + "*"*self.getCurrentTCD(2) + " : " + "TEST MESSAGE as Message is None" + "\n"
	                self.__writeLog(logString)

        def __writeLog(self,logString):
                try:
                        f = open(self.logfile,'a')
                except Exception as e:
                        print("Unable to open file for logging, exception : " + str(e))
			raise(Exception("Unable to open file for logging, exception : " + str(e)))
                f.write(logString)
                f.close()

	def globalprint(self,printdict):
	        depth=1
	        finalOutput = "{\n"
	        for key in printdict.keys():
	                depth += 1
	                finalOutput += "\t"*depth + str(key) + ": {\n"
	                for key2 in printdict[key].keys():
	                        depth += 1
	                        finalOutput += "\t"*depth + str(key2) + " : " + str(printdict[key][key2]) + "\n"
	                        depth -= 1
	                finalOutput += "\t"*depth + "}\n"
	                depth -= 1
	        depth -= 1
	        finalOutput += "}\n"
	        return finalOutput

	def printdict(self,mydict,depth=1,finalOutput=None,stringAware=False):
	        if depth == 1:
	                finalOutput = ""
	        if isinstance(mydict,dict):
	                finalOutput += "\t"*depth + "{\n"
	                depth += 1
	                for key in sorted(mydict.keys()):
	                        if isinstance(mydict[key],dict):
					if stringAware:
						finalOutput += "\t"*depth + str("'"+key+"'") + " : \n"
					else:
		                                finalOutput += "\t"*depth + str(key) + " : \n"
	                                finalOutput = self.printdict(mydict[key],depth,finalOutput,stringAware)
	                        else:
					if stringAware:
						if isinstance(mydict[key],str):
							finalOutput += "\t"*depth + str("'"+key+"'") + " : " + str("'"+mydict[key]+"'") + "," + "\n"
						else:
							finalOutput += "\t"*depth + str("'"+key+"'") + " : " + str(mydict[key]) + "," + "\n"
					else:
		                                finalOutput += "\t"*depth + str(key) + " : " + str(mydict[key]) + "\n"
	                depth -= 1
			if stringAware and depth != 1:
				finalOutput += "\t"*depth + "},\n"		
			else:
		                finalOutput += "\t"*depth + "}\n"
	                return finalOutput

	def dropColumns(self):
		tableColumnDropDict = dict()
		for key in self.globalColumnModificationDict.keys():
			for key2 in self.globalColumnModificationDict[key].keys():
				table = key2[0]
				try:
					column = int(key2[1].split("__")[1])
					if table in tableColumnDropDict.keys():
						if column in tableColumnDropDict[table]:
							tableColumnDropDict[table].remove(column)
							print("From table = %s removed column number = %d" %(table,column))
						else:
							print("From table = %s unable to remove column number = %d" %(table,column))
							print("Possible duplicate entry for column %s" %(str(key2[1])))
					else:
						tableColumnDropDict[table] = range(0,800)
						if column in tableColumnDropDict[table]:
							tableColumnDropDict[table].remove(column)
							print("From table = %s removed column number = %d" %(table,column))
						else:
							print("From table = %s unable to remove column number = %d" %(table,column))
							print("This strange, just created this list and it should have this column!!!!!")
				except Exception as e:
					print("base key = %s & sub key = %s and value = %s" %(str(key),str(key2),str(self.globalColumnModificationDict[key][key2])))
					print("Exception found in removing columns : %s" %(str(e)))
					continue
		return tableColumnDropDict

## Singleton Object initiation ##

def __globalinit():
        return ntiDSMGlobal()

global ntiDSMGlobalObj
ntiDSMGlobalObj = __globalinit()
