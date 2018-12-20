"""
DESCRIPTION: This script runs an exahustive search through all possible feature
             sets that could be used in conjunction with a classifier to classify
             all Event-Context pairs in a Biomedical paper. Note that we have
             identified 14 features and thus the exhaustive search must test
             16,383 different combinations. To allow the script to run as
             quickly as possible we have parallelized our script with MPI so
             that the running of this script will take advantage of all CPU
             resources it recieves. Please consider running this script with a
             large number of processing units.

SCRIPT PARAMETERS: classifier identifier for chosen classifier

SCRIPT OUTPUT: JSON file storing a list of all prediciton count dictionaries
               associated to their matching feature set. The prediction count
               dictionaries are made on a per-paper basis as the optimal
               feature set for a single classifier may differ from paper-to-paper
"""
from os.path import join
import json
import time
import sys
import os

import sklearn.metrics as metrics
from mpi4py import MPI

import utils.classifiers as clfs
import utils.folds as folds
import utils.utils as utils

COMM = MPI.COMM_WORLD


def split(container, count):
    """
    Simple function splitting a container into equal length chunks.

    Order is not preserved but this is potentially an advantage depending on
    the use case.
    """
    return [container[_i::count] for _i in range(count)]


if COMM.rank == 0:
    if len(sys.argv) != 2:
        raise RuntimeError("Incorrect number of input args: {}".format(len(sys.argv)))

    # Load features file to access original feature names
    json_config = utils.Config()
    data_path = json_config.get_features_filepath()
    feature_path = os.path.join(data_path, "features.csv")

    print("Reading initial data frame for features list")
    df = utils.load_dataframe(feature_path)
    data_features = utils.data_features_only(df)

    # Load grouped features data for classification
    print("Reading grouped dataframe")
    groups_path = os.path.join(data_path, "grouped_features.csv")
    df = utils.load_dataframe(groups_path)

    # Remove degenerate paper
    gdf2 = df[df.PMCID != "b'PMC4204162'"]

    # Step 2 - Create list of paper labels for fold generation
    print("Making paper-label lists and folds")
    row_labels = [pmc_id for pmc_id in gdf2["PMCID"].values]
    paper_labels = list(set(row_labels))
    fold_path = join(json_config.get_features_filepath(), "cv_folds_val_4.pkl")
    cv_folds = utils.load_cv_folds(fold_path)
    clf_type = sys.argv[1]

    # Step 3 - Get power set of features list
    print("Getting power set of features")
    feat_pow_set = utils.feature_power_set(data_features)
    feat_pow_set = feat_pow_set[:4]

    jobs = split(list(enumerate(feat_pow_set)), COMM.size)
else:
    gdf2 = None
    paper_labels = None
    cv_folds = None
    clf_type = None
    jobs = []

# Need to have shared access to these variables across all cores
(gdf2, paper_labels,
 cv_folds, clf_type) = COMM.bcast((gdf2, paper_labels, cv_folds, clf_type), root=0)

# Scatter jobs across cores.
jobs = COMM.scatter(jobs, root=0)

results = []
for job in jobs:
    # Step 4 - Gather prediction scores for all feature sets
    (f_idx, features) = job

    clf_type = sys.argv[1]
    classifier = clfs.select_classifier(clf_type, features, cv_folds)

    print("STARTING Feature set #{}".format(f_idx))
    start = time.time()
    per_paper_results = dict()
    per_paper_validation = dict()
    for idx, (train_set, validation_set, test_set) in enumerate(cv_folds):
        # 1. Perform training and fitting of classifier given curr features
        ((y_true, y_pred),
         (y_true_val, y_val)) = folds.fold_predictions(classifier, train_set,
                                                       test_set, validation_set,
                                                       gdf2, features)

        # Record scores from test set
        confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
        per_paper_results[paper_labels[idx]] = {
            "FP": int(confusion_matrix[0][1]),
            "FN": int(confusion_matrix[1][0]),
            "TP": int(confusion_matrix[1][1]),
            "TN": int(confusion_matrix[0][0])
        }

        # Record scores from validation set
        confusion_matrix = metrics.confusion_matrix(y_true_val, y_val)
        per_paper_validation[paper_labels[idx]] = {
            "FP": int(confusion_matrix[0][1]),
            "FN": int(confusion_matrix[1][0]),
            "TP": int(confusion_matrix[1][1]),
            "TN": int(confusion_matrix[0][0])
        }

    result = {
        "features": features,
        "by_paper_test": per_paper_results,
        "by_paper_validation": per_paper_validation
    }

    end = time.time()
    print("FINISHED Feature set #{}\t\t({}s)".format(f_idx, end - start))
    results.append(result)

# Gather results on rank 0.
results = COMM.gather(results, root=0)

if COMM.rank == 0:
    # Flatten list of lists.
    print("Outputting results to JSON")
    scores = [_i for temp in results for _i in temp]

    # Step 5 - Output scores to json
    with open("../data/{}_ablation_results.json".format(clf_type), "w") as outfile:
        json.dump(scores, outfile)
