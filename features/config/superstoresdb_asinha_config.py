config = {
	"entities" : {
		"cust_dimen" : {
			"feature_metadata" : {
				"Customer_Segment" : {
					"categorical" : True,
					"categorical_filter" : True,
					"numeric" : False
				}
			},
			"one-to-one" : [],
		},
		"market_fact" : {
			"feature_metadata" : {
				"Prod_id" : {
					"categorical" : True,
					"categorical_filter" : True,
					"numeric" : False
				}
			}
		}
	},
	"max_categorical_filter" : 10,
}
