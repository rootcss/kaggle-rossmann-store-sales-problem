#!/usr/bin/python

import numpy as np
import xgboost as xgb
import operator
from sklearn.cross_validation import train_test_split

class XGBoost:

	xgbModel = None
	xgb_params = {
		"objective": "reg:linear",
		"booster" : "gbtree",
		"eta": 0.09,
		"max_depth": 10,
		"subsample": 0.9,
		"colsample_bytree": 0.7,
		"silent": 1,
		"seed": 1301
	}
	xgb_num_boost_round = 10
	fmap_path = 'data/xgb.fmap'
	
	def __init__(self):
		pass

	def trainXGBModel(self, train, features):
		print("Training an XGB Model..")
		# Reference: http://xgboost.readthedocs.org/en/latest/python/python_intro.html
		X_train, X_valid = train_test_split(train, test_size=0.012, random_state=10)
		dmtrain = xgb.DMatrix(X_train[features], np.log1p(X_train.Sales))
		dmvalid = xgb.DMatrix(X_valid[features], np.log1p(X_valid.Sales))
		watchlist = [(dmtrain, 'train'), (dmvalid, 'eval')]
		self.xgbModel = xgb.train( \
							self.xgb_params, \
							dmtrain, \
							self.xgb_num_boost_round, \
							evals=watchlist, \
							early_stopping_rounds=100, \
							feval=self.rmspe_xg, \
							verbose_eval=True \
						)
		return {'X_train' : X_train, 'X_valid' : X_valid, 'xgbModel' : self.xgbModel}

	def validateXGBModel(self, model_data):
		print("Validating the XGB Model..")
		yhat = model_data['xgbModel'].predict(xgb.DMatrix(model_data['X_valid'][self.features]))
		error = self.rmspe(model_data['X_valid'].Sales.values, np.expm1(yhat))
		print('RMSPE: {:.6f}'.format(error))

	def predictUsingXGB(self, dataset, features):
		print("Predicting using XGB on the dataset..")
		return self.xgbModel.predict(xgb.DMatrix(dataset[features]))

	def calculateXGBFeatureImportances(self, features):
		self.generate_feature_map(features)
		importance = self.xgbModel.get_fscore(fmap=self.fmap_path)
		importance = sorted(importance.items(), key=operator.itemgetter(1))
		return importance

	def generate_feature_map(self, features):
		fp = open(self.fmap_path, 'w')
		for i, feat in enumerate(features):
		    fp.write('{0}\t{1}\tq\n'.format(i, feat))
		fp.close()
		print("XGB Map created at: " + self.fmap_path)

	def rmspe_xg(self, yhat, y):
		y = np.expm1(y.get_label())
		yhat = np.expm1(yhat)
		return "rmspe", self.rmspe(y,yhat)

	def rmspe(self, y, yhat):
		return np.sqrt(np.mean((yhat/y-1) ** 2))