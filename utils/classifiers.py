from os.path import join
from math import sqrt
import sys

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn import metrics
from tqdm import tqdm
import numpy as np

import utils.balancer as balancer
import utils.utils as utils

# NOTE: The following functions return various classifiers from scikit-learn.
# While we experimented with hyper-parameter values for our own experiments, our
# results can be extend by performing an actual GridSearch over the different
# parameters defined here. Do note that performing a GridSearch while also
# searching exhaustively over all feature sets is likely computationally infeasible


# =============================================================================
# LEARNING CLASSIFIER CODE
# =============================================================================
# Logisitic Regression classifier
def get_logistic_regression(tolerance=0.001):
    # print(f"Running LogisticRegression with tolerance={tolerance}")
    return LogisticRegression(penalty="l2", tol=tolerance)


# SVM -- Linear kernel classifier
def get_linear_svm():
    # print("Running SVM with linear kernel")
    return LinearSVC(C=0.001, penalty='l2', tol=0.001)


# SVM -- Polynomial kernel classifier
def get_poly_svm(degree=3, g=0.0001, C=10):
    #print(f"Running Poly kernel SVM with degree = {degree}, gamma={g}, C={C}")
    return SVC(kernel="poly", degree=degree, gamma=g, C=C)


# SVM -- Gaussian kernel classifier
def get_RBF_svm(C=500, gamma=.00008, tolerance=0.0001):
    # print(f"Running RBF kernel SVM with C = {C} and gamma = {gamma}, tolerance = {tolerance}")
    return SVC(kernel="rbf", C=C, gamma=gamma, tol=tolerance)


# Feed-forward Neural Network classifier
def get_ff_nn(num_feats, num_train, alpha=1, a=6, num_hidden=1, amt_first=.5, amt_second=.5):
    # Selecting the maximum number of nodes so that our network does not fit
    # exactly to the training data to avoid overfitting. NOTE that this is a
    # function of the number of features used in the classifier and the number
    # of training examples
    num_nodes = num_train // (a * (num_feats + 1))

    if num_hidden == 1:
        layers = (num_nodes, )
    elif num_hidden == 2:
        layers = (int(num_nodes * amt_first), int(num_nodes * amt_second))
    else:
        raise ValueError("Too many hidden layers")

    # print(f"Running Feed-forward Neural Net with L2 reg = {alpha} and hidden layers = {layers}")
    return MLPClassifier(hidden_layer_sizes=layers, alpha=alpha, max_iter=500)


# Random Forest classifier
def get_random_forest(num_feats, depth=8):
    est = 32
    empirical_max = int(sqrt(num_feats))
    # print("Running RandomForest with max_depth = {}, n_estimators = {}".format(depth, est))
    return RandomForestClassifier(max_depth=depth,
                                  n_estimators=est,
                                  max_features=empirical_max)


def get_grad_tree_classifier():
    return GradientBoostingClassifier(random_state=0,
                                      warm_start=True,
                                      n_estimators=300)
# =============================================================================


# =============================================================================
# CLASSIFIER SELECTOR
# =============================================================================
def select_classifier(clf_type, features, folds):
    if clf_type == "log_reg":
        classifier = get_logistic_regression()
    elif clf_type == "linear_svm":
        classifier = get_linear_svm()
    elif clf_type == "poly_svm":
        classifier = get_poly_svm()
    elif clf_type == "rbf_svm":
        classifier = get_RBF_svm()
    elif clf_type == "neural_net":
        classifier = get_ff_nn(len(features), len(folds[0][0]))
    elif clf_type == "rand_forest":
        classifier = get_random_forest(len(features))
    elif clf_type == "grad_boost":
        classifier = get_grad_tree_classifier()
    else:
        raise RuntimeError("Unrecognized clf-type input arg: {}".format(clf_type))

    return classifier
# =============================================================================


# =============================================================================
# BASELINE CLASSIFIER CODE
# =============================================================================
def baseline_counts(feat_path):
    """
    Given a path to a grouped features csv file, this function loads the
    features and uses the deterministic baseline classifier to predict the
    labels for each row. Predicions are then returned in the form of T/F P/N counts

    :feat_path: A string specifying the absolute path to grouped_features.csv
    :return: A dictionary holding all T/F, P/N counts
    """
    # Step 1 - Load the dataframe that has been pre-balanced and pre-grouped, prepare output file
    df_path = join(feat_path, "grouped_features.csv")
    print("Loading data from: {}".format(df_path))
    gdf = utils.load_dataframe(df_path)
    gdf2 = gdf[gdf.PMCID != "b'PMC4204162'"]

    # Step 2 - Create list of paper labels for fold generation
    print("Loading CV folds")
    fold_path = join(feat_path, "cv_folds_val_4.pkl")
    fold_iter = utils.load_cv_folds(fold_path)

    # Step 3 - Calculate baseline score for each fold
    truth_values = list()
    predicted_values = list()
    print("Scoring testing data using deterministic sentence distance")
    for (train, _, test) in tqdm(list(fold_iter), desc="Testing over folds"):
        train_df = gdf2.iloc[train][["PMCID", "sentenceDistance_min", "label"]]
        test_df = gdf2.iloc[test][["sentenceDistance_min", "label"]]

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

        # save predicitons and truth labels for later scoring
        truth_values.extend(y_true)
        predicted_values.extend(y_pred)

    confusion_matrix = metrics.confusion_matrix(truth_values, predicted_values)

    return {
        "FP": int(confusion_matrix[0][1]),
        "FN": int(confusion_matrix[1][0]),
        "TP": int(confusion_matrix[1][1]),
        "TN": int(confusion_matrix[0][0])
    }


def per_fold_baseline_counts(return_counts=False):
    """
    This function runs our deterministic baseline classifier over each fold
    (paper) in our dataset. The baseline classifier predictions per-paper are
    scored via the F1 metric.

    :return: A dictionary of baseline F1 scores per paper
    """
    # Step 1 - Load the dataframe that has been pre-balanced and pre-grouped, prepare output file
    config = utils.Config() if len(sys.argv) <= 1 else utils.Config(sys.argv[1])
    feature_path = join(config.get_features_filepath(), "grouped_features.csv")
    print("Loading data from: {}".format(feature_path))
    gdf = utils.load_dataframe(feature_path)
    gdf2 = gdf[gdf.PMCID != "b'PMC4204162'"]

    # Step 2 - Create list of paper labels for fold generation
    print("Loading CV folds")
    fold_path = join(config.get_features_filepath(), "cv_folds_val_4.pkl")
    fold_iter = utils.load_cv_folds(fold_path)

    # Step 3 - Calculate baseline score for each fold
    f1_scores = dict()
    all_counts = dict()
    print("Scoring testing data using deterministic sentence distance")
    for (train, _, test) in tqdm(list(fold_iter), desc="Testing over folds"):
        train_df = gdf2.iloc[train][["PMCID", "sentenceDistance_min", "label"]]
        test_df = gdf2.iloc[test][["PMCID", "sentenceDistance_min", "label"]]

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

        # save predicitons and truth labels for later scoring
        confusion_matrix = metrics.confusion_matrix(y_true, y_pred)

        counts = {
            "FP": int(confusion_matrix[0][1]),
            "FN": int(confusion_matrix[1][0]),
            "TP": int(confusion_matrix[1][1]),
            "TN": int(confusion_matrix[0][0])
        }

        fold_f1 = utils.f1(counts)
        pmcid_label = list(test_df["PMCID"].values)[0]
        f1_scores[pmcid_label] = fold_f1
        all_counts[pmcid_label] = counts

    if return_counts:
        return all_counts
    else:
        return f1_scores
# =============================================================================
