from sklearn.ensemble import GradientBoostingClassifier
from os.path import join
import xgboost as xgb
import os
from multiprocessing import Pool, cpu_count
import sklearn.metrics as metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

import utils.utils as utils
# import utils.folds as folds
#import utils.cross_val as cv
#from utils.config import Config
import utils.classifiers as clfs
import utils.folds as cv

json_config = utils.Config()
data_path = json_config.get_features_filepath()
feature_path = os.path.join(data_path, "features.csv")

print("Reading grouped dataframe")
groups_path = os.path.join(data_path, "grouped_features.csv")
df = utils.load_dataframe(groups_path)
gdf2 = df[df.PMCID != "b'PMC4204162'"]
data_features = utils.data_features_only(gdf2)
# Step 2 - Create list of paper labels for fold generation
print("Making paper-label lists and folds")
row_labels = [pmc_id for pmc_id in gdf2["PMCID"].values]
paper_labels = list(set(row_labels))
fold_path = join(json_config.get_features_filepath(), "cv_folds_val_4.pkl")
cv_folds = utils.load_cv_folds(fold_path)
#cv_folds = [(train + validate, test) for (train, validate, test) in cv_folds]
cv_folds = [(train, validate, test) for (train, validate, test) in cv_folds]
clf = GradientBoostingClassifier(random_state=0, warm_start=True, n_estimators=300)

giant_test_label = list()
giant_pred_test_label = list()
"""
for idx, (train_set, test_set) in enumerate(cv_folds):
 	(y_test, y_pred) = cv.combined_train_val_fold_predictions(clf, train_set,
                                                       test_set,
                                                       gdf2, data_features)
 	giant_test_label.extend(y_test)
 	giant_pred_test_label.extend(y_pred)

"""

for idx, (train_set, validate_set, test_set) in enumerate(cv_folds):
	((y_test,y_pred), (y_test_val, y_pred_val)) = cv.fold_predictions(clf, train_set, test_set, validate_set, gdf2, data_features)
	giant_test_label.extend(y_test)
	giant_pred_test_label.extend(y_pred)

print("Scoring our predictions")
micro_array = dict()
f1_micro = metrics.f1_score(giant_test_label, giant_pred_test_label)
recall_micro = metrics.recall_score(giant_test_label, giant_pred_test_label)
precision_micro = metrics.precision_score(giant_test_label, giant_pred_test_label)
accuracy_micro = metrics.accuracy_score(giant_test_label, giant_pred_test_label)
micro_array["micro_f1_score"] = f1_micro
micro_array["micro_precision_score"] = precision_micro
micro_array["micro_recall_score"] = recall_micro
micro_array["micro_accuracy_score"] = accuracy_micro
print(micro_array)