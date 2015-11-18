#!/usr/bin/python

#%matplotlib inline
import os
from core.rossmann import Rossmann

core_files = {
	'train' : "data/train.csv",
	'test' : "data/test.csv",
	'store' : "data/store.csv",
	'submission_file' : "output/submission.csv"
}
ross = Rossmann(core_files)
ross.loadDataSets()
ross.dataFilterAndCleaning()

# Preparing to train data
ross.develop_features()
unique_features = []
for feature in ross.features:
    if feature not in unique_features:
        unique_features.append(feature)
ross.features = unique_features
ross.xgb_num_boost_round = 1
model_data = ross.trainXGBModel(ross.train, ross.features)

# Validating and Printing
ross.validateXGBModel(model_data)
test_probs = ross.predictUsingXGB(ross.test, ross.features)

# Submitting
ross.submission(test_probs)

# Calculating & plotting Feature Importance
ross.fmap_path = 'output/xgb.fmap'
imp = ross.calculateXGBFeatureImportances(ross.features)
ross.plotting(imp, output='output/plot_feature_importance_using_xgb.png')