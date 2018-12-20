import csv
from os.path import join
import os
import time
from multiprocessing import Pool, cpu_count
import sklearn.metrics as metrics
from tqdm import tqdm
import sys
import json

import utils.utils as utils
import utils.folds as cv
import utils.classifiers as clfs

json_config = utils.Config()
data_path = json_config.get_features_filepath()
preds_dict = os.path.join(data_path, "feature_set_comp/")

json_list = [fname for fname in os.listdir(preds_dict) if fname.endswith(".json")]

with open("feature_comparison_results.csv", "w", newline = '') as csvfile:
	table_columns = ["Classifier", "Feature set", "P", "R", "F1"]
	writer = csv.writer(csvfile, delimiter = '\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
	writer.writerow(table_columns)
	feat_set_list = None
	for classifier in json_list:
		current_json = json.load(open(join(preds_dict, classifier), "r"))
		
		classifer_name = utils.resolve_model_name(classifier)

		if feat_set_list is None:
			feat_set_list = sorted(list(current_json.keys()))
		
		for feature_set in feat_set_list:
			pred_dict = current_json[feature_set]
			row = list()
			row.append(classifer_name)
			row.append(feature_set)
			f1 = utils.f1(pred_dict)
			p = utils.precision(pred_dict)
			r = utils.recall(pred_dict)
			row.append(p)
			row.append(r)
			row.append(f1)
			writer.writerow(row)



		



