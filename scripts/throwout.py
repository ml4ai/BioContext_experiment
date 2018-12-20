"""
DESCRIPTION: This script was used to verify the F1 score of the one paper from
             our original dataset that we choose to exclude from our final
             results. This paper recieves an F1 score of lower than 0.1 for
             every classifier and thus is an extreme outlier.

SCRIPT PARAMETERS: None

SCRIPT OUTPUT: P/R/F1 scores for baseline and learning classifiers on the thrown
               out paper.
"""
from os.path import join
import json

import sklearn.metrics as metrics
from tqdm import tqdm
import numpy as np

import utils.balancer as balancer
import utils.classifiers as clf
import utils.folds as folds
import utils.utils as utils


json_config = utils.Config()
DRIVE_PATH = json_config.get_features_filepath()
ablation_path = json_config.get_ablation_path()
feature_path = join(DRIVE_PATH, "grouped_features.csv")
gdf = utils.load_dataframe(feature_path)
print("Loaded the dataframe")

throwout_id = "b'PMC4204162'"
row_labels = [pmc_id for pmc_id in gdf["PMCID"].values]
paper_labels = list(set(row_labels) - set([throwout_id]))
np.random.shuffle(paper_labels)
training_labels = paper_labels[:17]
validation_labels = paper_labels[17:21]

paper_groups = gdf.groupby("PMCID")

training_rows = list()
validation_rows = list()
testing_rows = list()
for pid, frame in tqdm(paper_groups, desc="Iterating over papers"):
    if pid == throwout_id:
        for idx, row in frame.iterrows():
            testing_rows.append(idx)
    elif pid in training_labels:
        for idx, row in frame.iterrows():
            training_rows.append(idx)
    elif pid in validation_labels:
        for idx, row in frame.iterrows():
            validation_rows.append(idx)


# BEGIN BASELINE
train_df = gdf.iloc[training_rows][["PMCID", "sentenceDistance_min", "label"]]
test_df = gdf.iloc[testing_rows][["PMCID", "sentenceDistance_min", "label"]]

# Need to correct or class imbalance on the training data
train_balanced_df = balancer.balance_by_paper(train_df, 1)

k_f1_scores = list()
for k_val in range(51):
    (score_dict, _, _) = utils.deterministic_sent_dist(train_balanced_df, k=k_val)

    # record the F1 score for this value of k
    k_f1_scores.append(score_dict["f1_score"])

# Select the value of k which corresponds to the max F1 score
best_k = np.argmax(k_f1_scores)

# get predicted values from test set using the trained value of k
(_, y_true, y_pred) = utils.deterministic_sent_dist(test_df, k=best_k)
f1 = metrics.f1_score(y_true, y_pred)
p = metrics.precision_score(y_true, y_pred)
r = metrics.recall_score(y_true, y_pred)
print("Baseline\tP: {}\tR: {}\tF1:{}".format(p, r, f1))
# END BASELINE

classifiers = ["log_reg", "linear_svm", "rbf_svm", "poly_svm", "rand_forest", "neural_net"]

results = list()
for clf_type in classifiers:
    json_filename = "{}/{}_ablation_results.json".format(ablation_path, clf_type)
    feature_data = json.load(open(json_filename))

    paper_scores = {pid: [] for pid in feature_data[0]["by_paper_validation"].keys()}
    for feat_dict in tqdm(feature_data, desc="Processing {}".format(clf_type)):
        for pid, scores in feat_dict["by_paper_validation"].items():
            f1 = utils.f1(scores)
            paper_scores[pid].append((f1, scores,
                                      feat_dict["by_paper_test"][pid],
                                      feat_dict["features"]))

    def best_score(scores):
        (f1, val_preds, test_preds, feats) = max(scores, key=lambda tup: tup[0])
        return (f1, feats)

    best_paper_scores = {pid: best_score(scores) for pid, scores in paper_scores.items()}
    paper_scores = list(best_paper_scores.values())
    np.random.shuffle(paper_scores)
    (_, features) = paper_scores[0]

    if clf_type == "log_reg":
        classifier = clf.get_logistic_regression()
    elif clf_type == "linear_svm":
        classifier = clf.get_linear_svm()
    elif clf_type == "poly_svm":
        classifier = clf.get_poly_svm()
    elif clf_type == "rbf_svm":
        classifier = clf.get_RBF_svm()
    elif clf_type == "neural_net":
        classifier = clf.get_ff_nn(len(features), len(training_rows))
    elif clf_type == "rand_forest":
        classifier = clf.get_random_forest(len(features))
    else:
        raise RuntimeError("Unrecognized clf-type input arg: {}".format(clf_type))
    ((y_t_t, y_p_t), (y_t_v, y_p_v)) = folds.fold_predictions(classifier, training_rows, testing_rows, validation_rows, gdf, features)
    f1 = metrics.f1_score(y_t_t, y_p_t)
    p = metrics.precision_score(y_t_t, y_p_t)
    r = metrics.recall_score(y_t_t, y_p_t)
    results.append((clf_type, p, r, f1))

print(results)
