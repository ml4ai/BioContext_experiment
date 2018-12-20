from itertools import combinations, chain
from os.path import normpath
import pickle
import json

from sklearn import metrics
import pandas


# =============================================================================
# SCORES FROM PREDICTION DICTS
# =============================================================================
def f1(preds):
    """
    Given a prediction dictionary, this function returns the F1 score of the
    prediction scores

    :preds: A dictionary with T/F P/N counts
    :return: A floating point F1 score w/range --> [0, 1]
    """
    p = precision(preds)
    r = recall(preds)
    if p + r == 0:
        return 0
    return 2 * (p * r) / (p + r)


def precision(preds):
    """
    Given a prediction dictionary, this function returns the precision score of
    the prediction scores

    :preds: A dictionary with T/F P/N counts
    :return: A floating point precision score w/range --> [0, 1]
    """
    if preds["TP"] + preds["FP"] == 0:
        return 0

    return preds["TP"] / (preds["TP"] + preds["FP"])


def recall(preds):
    """
    Given a prediction dictionary, this function returns the recall score of the
    prediction scores

    :preds: A dictionary with T/F P/N counts
    :return: A floating point recall score w/range --> [0, 1]
    """
    if preds["TP"] + preds["FN"] == 0:
        return 0
    return preds["TP"] / (preds["TP"] + preds["FN"])
# =============================================================================


# =============================================================================
# CONFIGURATION HELPER
# =============================================================================
# Used to access the config.json file that will allow users to set personal
# system paths to access features.csv and ablation data files
class Config():
    def __init__(self, path="../config.json"):
        self.config_file = path
        self.data = json.load(open(self.config_file))

    def get_features_filepath(self):
        return self.data["features_path"]

    def get_ablation_path(self):
        return self.data["ablation_path"]
# =============================================================================


# ==============================================================================
# BASELINE SCORE CALCULATION
# ==============================================================================
def deterministic_sent_dist(dataframe, k=7):
    """
    Given a DataFrame object, this function returns the P/R/F1 scores for each
    of the rows present

    :DataFrame: The DataFrame containing all rows to classify
    :k: The sentenceDistance range to use for searching

    :return: score dictionary with P/R/F1 scores
    """
    truth_values = list(dataframe['label'])
    predicted_values = list(dataframe['sentenceDistance_min'] <= k)

    micro_array = {
        "f1_score": metrics.f1_score(truth_values, predicted_values),
        "precision_score": metrics.precision_score(truth_values, predicted_values),
        "recall_score": metrics.recall_score(truth_values, predicted_values),
        "accuracy_score": metrics.accuracy_score(truth_values, predicted_values),
    }

    return (micro_array, truth_values, predicted_values)
# ==============================================================================


# =============================================================================
# PIPELINE CODE
# =============================================================================
def load_dataframe(path):
    """
    Takes a filepath (in UNIX form) to a saved csv file that we use for
    storing a Pandas dataframe. This function will normalize the path variable
    so that it can be used on any operating system

    :path: A string that represents a filepath to a csv file
    :return: The dataframe contained in the csv file
    """
    return pandas.read_csv(normpath(path))


def load_cv_folds(filepath):
    """
    Given a path to a pickle file that contains a set of premade cross-validation
    folds, this function will load the file and return the contained data. This
    function also performs simple checks on the object loaded to be sure it is
    a well-formed CV-folds object

    :filepath: A string that specifies the absolute path to a saved folds object
    :return: A list where each object is a fold for CV of the form
             (train, validation, test) each of which is a list of indicies of
             elements from grouped_features.csv to use in cross validation
    """
    folds = pickle.load(open(filepath, "rb"))

    if not isinstance(folds, list):
        raise RuntimeError("Loaded a non-list item as CV folds.")

    if not isinstance(folds[0], tuple) or not len(folds[0]) == 3:
        print(type(folds[0]))
        print(len(folds))
        raise RuntimeError("CV-folds object is malformed")

    return folds


def make_feature_label_sets(df, features):
    """
    Creates a feature matrix X and a labels vector y from a dataframe.
    :df: A pandas dataframe containing all data to be used in X and y
    :features: A list of features that represent columns in df. These features
               are the data that will be included from df in X
    :return: X, y where X is a numpy matrix of features and y is a numpy array
             of boolean values
    """
    X = df[features]
    X = X.values if len(features) > 1 else X.values.reshape((X.size, 1))

    y = df['label'].values.astype("bool")

    return X, y


def resolve_model_name(filename):
    """
    Given the filename from an ablation file this function parsers out the
    identifier of the classifier used and then returns the print name which
    matches the identifier

    :filename: A string from the ablationstudy results JSON file
    :return: A string representing the print name of a classifier
    """
    first_ = filename.find("_")
    second_ = filename.find("_", first_ + 1)
    model_name = filename[:second_]
    return get_classifier_print_name(model_name)


def get_classifier_print_name(model_name):
    if model_name == "log_reg":
        return "Logistic Reg."
    elif model_name == "linear_svm":
        return "SVM - Linear"
    elif model_name == "poly_svm":
        return "SVM - Poly"
    elif model_name == "rbf_svm":
        return "SVM - Gaussian"
    elif model_name == "neural_net":
        return "Feed Fwd. NN"
    elif model_name == "rand_forest":
        return "Random Forest"
    elif model_name == "grad_boost":
        return "Grad. Tree Boost"
    else:
        raise RuntimeError("Unrecognized model-type: {}".format(model_name))
# =============================================================================


# =============================================================================
# FEATURE SET MANIPULATION
# =============================================================================
def data_features_only(df):
    """
    Given a pandas dataframe from our features DataFrame set, this function
    returns the column names of all data features contained in the DataFrame

    :df: A pandas DataFrame object
    :return: A list of all data features
    """
    columns = list(df.columns.values)
    meta_cols = ["PMCID", "EvtID", "CtxID", "label", "Unnamed: 0"]

    return list(set(columns) - set(meta_cols))


def expanded_features(feats):
    """
    Given a list of features this function expands the list of features by
    creating a min, max, avg feature for each original feature. This is used to
    retain data from our features when creating a row grouping to represent a
    single collection of Event-Context pairs.

    :feats: A list of strings where each string is a feature name
    :return: A list of strings for each of the new features
    """
    results = list()
    for feat in feats:
        results.extend(["{}_min".format(feat),
                        "{}_avg".format(feat),
                        "{}_max".format(feat)])

    return results


def feature_power_set(data_features):
    """
    Given a list of features, this function returns a list of all possible
    combinations of the features. This is akin to taking the power set of the
    original feature list; however, we make an acception to group all
    context-dependency-tail features into a single feature and all
    event-dependency-tail features into a single feature as either all or none
    of the data contained in each of these will need to be used to be meaningful
    for classification.

    :data_features: A list of strings that represent all features
    :return: A list of list representing all possible combinations of features
    """
    # Find all context-dep-tail/event-dep-tail features
    ctx_dep_cols = [c for c in data_features if "ctxDepTail" in c]
    evt_dep_cols = [c for c in data_features if "evtDepTail" in c]

    # Remove dep-tail features from overall list
    reg_cols = list(set(data_features) - set(ctx_dep_cols + evt_dep_cols))

    # Add lists of dep-tail features as single elements
    reg_cols.append(ctx_dep_cols)
    reg_cols.append(evt_dep_cols)

    # Finds the power set of all features in the cleaned version of data_features
    pow_set = chain.from_iterable(combinations(reg_cols, r)
                                  for r in range(len(reg_cols)+1))

    # Returns the grouped stat variant of a feature
    def get_feature_stats(f):
        return [f + "_min", f + "_avg", f + "_max"]

    # Flatten lists in power set so that feature sets that include dep-tail
    # features do not have a nested list as a member of their feature set
    expanded_pow_set = list()
    for feat_set in pow_set:
        if len(feat_set) > 0:
            new_feat_set = list()
            for item in feat_set:
                if isinstance(item, list):
                    for feat in item:
                        new_feat_set.extend(get_feature_stats(feat))
                else:
                    new_feat_set.extend(get_feature_stats(item))

            expanded_pow_set.append(new_feat_set)

    return expanded_pow_set


def get_original_features(features):
    """
    Given an expanded set of features, this function returns the list of
    features that are contained in the original pandas DataFrame with the
    exception that dependency-tail features are represented as a single string.

    :features: a list of strings representing features
    :return: a list of strings of the original feature names
    """
    # Remove _max, _min, _avg, etc. endings and remove duplicates. (Duplicates
    # are caused by the removal of the endings)
    names = list(set([feat[:feat.rfind("_")] for feat in features]))

    # Group dep-tail features
    ctx_dep_cols = [c for c in names if "ctxDepTail" in c]
    evt_dep_cols = [c for c in names if "evtDepTail" in c]

    # Remove dep-tail features
    reg_names = list(set(names) - set(ctx_dep_cols + evt_dep_cols))

    # Add label for context-dep-tail features if any ctx-dep-tail features were
    # found in the original list of features
    if len(ctx_dep_cols) > 0:
        reg_names.append("CTX_DEP_TAIL_FEATS")

    # Add label for event-dep-tail features if any evt-dep-tail features were
    # found in the original list of features
    if len(evt_dep_cols) > 0:
        reg_names.append("EVT_DEP_TAIL_FEATS")

    return reg_names


# Code pertaining to the Oct 5th paper
def createFeaturesLists(data_features):
    ctx_dep_cols = [c for c in data_features if "ctxDepTail" in c]
    evt_dep_cols = [c for c in data_features if "evtDepTail" in c]
    reg_names = list(set(data_features) - set(ctx_dep_cols + evt_dep_cols))


    return {
        "All_features": data_features,
        "NDF": reg_names,
        "NDF_EVT": reg_names + evt_dep_cols,
        "NDF_CTX": reg_names + ctx_dep_cols,
        "CTX_EVT": ctx_dep_cols + evt_dep_cols

    }
# =============================================================================


# =============================================================================
# PAPER TYPE CHUNKING: UNUSED IN 2018 PAPER
# =============================================================================
def chunks_by_paper(df):
    paper_types = get_paper_type_dict()
    return paper_type_indexes(df, paper_types)


def get_paper_type_dict():
    results = dict()
    with open("../data/paper-types.txt", "r+") as types_file:
        for idx, line in enumerate(types_file):
            if idx > 1:
                fields = line.split(",")
                paper_id = fields[0].strip()
                paper_type = fields[1].strip()
                results[paper_id] = paper_type

    return results


def paper_type_indexes(features, id_2_class):
    survey_rows = list()
    discov_rows = list()
    print("Indexing by paper-type")
    for i, (idx, row) in enumerate(features.iterrows()):
        pmc_id = row["PMCID"].decode('utf-8')
        if pmc_id not in id_2_class:
            print("PMCID: {} was not found in ID dict!")
        else:
            if id_2_class[pmc_id] == "survey":
                survey_rows.append(i)
            elif id_2_class[pmc_id] == "discovery":
                discov_rows.append(i)
            else:
                print("Found row that does not match survey or discovery: {}".format(pmc_id))

    return {
        "survey": survey_rows,
        "discov": discov_rows
    }
# =============================================================================
