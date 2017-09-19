config = {
	"entities" : {
		"Profit_Outcome" : {
			"feature_metadata" : {
				"dropped_out" : {
					"categorical" : True,
					"numeric" : False,
					# "categorical_filter": True,
				}, 
			},
			"one-to-one" : ["cust_dimen"],
			"included_row_functions":[],
			"excluded_row_functions":[]
		},

		"prod_dimen" : {
			"feature_metadata" : {
			},
			"one-to-one" : [],
		},

		"orders_dimen" : {
			"feature_metadata" : {
			},
		},
           
		"shipping_dimen" : {
			"feature_metadata" : {
			},
		},
                  

		"cust_dimen" : {
			"feature_metadata" : {

				"test" : {
					"ignore": True
				}
			},
			'excluded_predict_entities' : ["Profit_Outcomes"]
			# "train_filter" : [["test", "=", 0]],
		},

		"market_fact" : {
		#	"feature_metadata" : {
				# "source" : {
				# 	"categorical" : True,
				# 	"numeric" : False,
				# 	"categorical_filter" : True
				# }, 
				"test" : {
					"ignore" : True
				}
			},
		    "included_row_functions":[],
			# "train_filter" : [["test", "=", 0]],
		},

		
	},

	"max_categorical_filter" : 1,
}
