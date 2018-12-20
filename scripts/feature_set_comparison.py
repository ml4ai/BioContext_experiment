from os.path import join
import os
import time
from multiprocessing import Pool, cpu_count
import sklearn.metrics as metrics
from tqdm import tqdm
import sys
import json

import utils.classifiers as clfs
import utils.utils as utils
import utils.folds as cv

# ==============================================================================
# BEGIN PREAMBLE
# ==============================================================================
print("Beginning preamble...")
clf_name = sys.argv[1]      # shortname of classifier to use
start_time = time.time()
json_config = utils.Config()
data_path = json_config.get_features_filepath()
feature_path = os.path.join(data_path, "features.csv")

# Load the Grouped DataFrame
groups_path = os.path.join(data_path, "grouped_features.csv")
df = utils.load_dataframe(groups_path)

# Remove degenerate paper
gdf2 = df[df.PMCID != "b'PMC4204162'"]
data_features = utils.data_features_only(gdf2)
features_dict_of_interest = utils.createFeaturesLists(data_features)

# Create list of paper labels for fold generation
row_labels = [pmc_id for pmc_id in gdf2["PMCID"].values]
paper_labels = list(set(row_labels))

# Load train/validate/test folds
fold_path = join(json_config.get_features_filepath(), "cv_folds_val_4.pkl")
cv_folds = utils.load_cv_folds(fold_path)

# Combining train and validation sets for each fold
cv_folds = [(train + validate, test) for (train, validate, test) in cv_folds]
print("Finished preamble. ({:.3f}s)".format(time.time() - start_time))
# ==============================================================================
# END PREAMBLE
# ==============================================================================


# TODO: finish writing this function
def score_feature_set(data):
    (cv_folds, df, new_features, set_name) = data
    giant_test_label = list()
    giant_pred_test_label = list()
    classifier = clfs.select_classifier(clf_name, new_features, cv_folds)
    for train, test in cv_folds:
        (y_test, pred_test) = cv.train_only_predictions(classifier, train, test, df, new_features)
        giant_test_label.extend(y_test)
        giant_pred_test_label.extend(pred_test)

    confusion_matrix = metrics.confusion_matrix(giant_test_label, giant_pred_test_label)

    return (set_name, {
        "FP": int(confusion_matrix[0][1]),
        "FN": int(confusion_matrix[1][0]),
        "TP": int(confusion_matrix[1][1]),
        "TN": int(confusion_matrix[0][0])
    })


# def predict_on_fold(data):
#     (fold, df, classifier, new_features) = data
#     (idx, (train, test)) = fold
#     return cv.train_only_predictions(classifier, train, test, df, new_features)


# max_cpus = cpu_count()
# num_cpus = 22 if max_cpus >= 22 else max_cpus
num_feat_sets = len(features_dict_of_interest.keys())
p = Pool(min(num_feat_sets, cpu_count()))
input_data = [(cv_folds, df, features, set_name)
              for set_name, features in features_dict_of_interest.items()]

res = list(tqdm(p.imap(score_feature_set, input_data), total=num_feat_sets, desc="Run FeatSets w/ {}".format(clf_name)))
results = {set_name: pred_dict for set_name, pred_dict in res}

# results = {set_name: None for set_name in features_dict_of_interest.keys()}
# for set_name, features in features_dict_of_interest.items():
#     classifier = clfs.select_classifier(clf_name, features, cv_folds)
#     giant_test_label = list()
#     giant_pred_test_label = list()
#
#     for (y_true, y_pred) in res:
#         giant_test_label.extend(y_true)
#         giant_pred_test_label.extend(y_pred)
#
#     confusion_matrix = metrics.confusion_matrix(giant_test_label, giant_pred_test_label)
#     results[set_name] = {
#         "FP": int(confusion_matrix[0][1]),
#         "FN": int(confusion_matrix[1][0]),
#         "TP": int(confusion_matrix[1][1]),
#         "TN": int(confusion_matrix[0][0])
#     }

with open(join(data_path, "{}_feature_set_comp.json".format(clf_name)), "w") as outfile:
    json.dump(results, outfile)
