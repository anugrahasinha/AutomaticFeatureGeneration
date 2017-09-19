

import ntiDSMGlobal as ntiGlb
from ntiDSMConfigGenerator import ntiDSMConfigGenerator
import ntiDSMMakeFeatures as ntiDSMEngine

class ntiDSM:

	def __init__(self):
		ntiGlb.ntiDSMGlobalObj.log("Starting with NTI DSM FRAMEWORK")
		try:
			ntiConfigGeneratorObj = ntiDSMConfigGenerator()
		except Exception as e:
			print("Problem while building config : " + str(e))
			ntiGlb.ntiDSMGlobalObj.log("Problem while building config : " + str(e))
			exit(255)
		ntiDSMEngine.ntiDSMStartFeatureGenEngine(ntiConfigGeneratorObj.databaseName,ntiConfigGeneratorObj.configFileName)





if __name__ == "__main__":
	ntiDSM()
