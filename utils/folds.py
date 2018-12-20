import numpy as np

import utils.balancer as balancer
import utils.utils as utils


def paper_fold_lists(row_paper_labels, rows, val_size=4):
    """
    This function takes a list of paper labels, one label per paper, and
    generates new train/test sets one fold at a time based upon the paper
    labels. Each new fold will have a single paper as the test set and all
    other data as the training set. This generator must keep track of which
    papers have been used as the testing sets to ensure each paper is used for
    testing. Train and test sets should be able to be shuffled if desired when
    this function is called. This function must yield a train/test set upon
    each iterator call.

    :row_paper_labels: A list of the paper labels, one label for paper
    :rows: A list of entries that show which index in the dataframe belongs to
           which paper. The values in this list are PMCIDs for a paper
    :shuffle_in_fold: A boolean flag to signal whether to shuffle train/test
                      sets before yielding.

    :return: a list of folds
    """
    testing_sets = {paper_id: list() for paper_id in row_paper_labels}

    for idx, paper_id in enumerate(rows):
        testing_sets[paper_id].append(idx)

    folds = list()
    for idx, pmcid in enumerate(row_paper_labels):
        testing_set = testing_sets[pmcid]

        other_ids = list(set(row_paper_labels) - set([pmcid]))
        np.random.shuffle(other_ids)
        validation_ids = other_ids[:val_size]
        training_ids = other_ids[val_size:]

        validation_set = list()
        training_set = list()
        for idx, paper_id in enumerate(rows):
            if paper_id in validation_ids:
                validation_set.append(idx)
            elif paper_id in training_ids:
                training_set.append(idx)

        folds.append((training_set, validation_set, testing_set))

    return folds


def train_only_predictions(classifier, train, test, df, new_features):
    """
    Given a classifier (train, val, test) sets for a fold, the original DataFrame
    and a set of features including truth labels, this function returns two
    tuples of the form (truth, predictions) for the (validation, testing) sets.

    :classifier: One of our scikit-learn learning classifiers
    :train: A set of indicies of rows in df to be used for training
    :test: A set of indicies of rows in df to be used for testing
    :validation: A set of indicies of rows in df to be used for validating
    :df: The original DataFrame that we will use to pull actual data from rows
    :new_features: The list of features we will use for fitting/predicting

    :return: nested 2-tuples of truth, predicted values
    """
    train_df = df.iloc[train]
    test_df = df.iloc[test]

    y_true = test_df["label"].values

    train_balanced_df = balancer.balance_by_paper(train_df, 1)

    X_train, y_train = utils.make_feature_label_sets(train_balanced_df, new_features)
    X_test, y_test = utils.make_feature_label_sets(test_df, new_features)

    classifier.fit(X_train, y_train)
    return ((y_true, classifier.predict(X_test)))


def combined_train_val_fold_predictions(clf, train,test,df, data_features):
    train_df = df.iloc[train]
    test_df = df.iloc[test]
    y_true = test_df["label"].values
    train_balanced_df = balancer.balance_by_paper(train_df, 1)
    X_train, y_train = utils.make_feature_label_sets(train_balanced_df, data_features)
    X_test, y_test = utils.make_feature_label_sets(test_df, data_features)
    clf.fit(X_train, y_train)
    return (y_true, clf.predict(X_test))

def fold_predictions(classifier, train, test, validation, df, new_features):
    """
    Given a classifier (train, val, test) sets for a fold, the original DataFrame
    and a set of features including truth labels, this function returns two
    tuples of the form (truth, predictions) for the (validation, testing) sets.

    :classifier: One of our scikit-learn learning classifiers
    :train: A set of indicies of rows in df to be used for training
    :test: A set of indicies of rows in df to be used for testing
    :validation: A set of indicies of rows in df to be used for validating
    :df: The original DataFrame that we will use to pull actual data from rows
    :new_features: The list of features we will use for fitting/predicting

    :return: nested 2-tuples of truh, predicted values
    """
    train_df = df.iloc[train]
    test_df = df.iloc[test]
    val_df = df.iloc[validation]
    y_true = test_df["label"].values
    y_true_val = val_df["label"].values

    train_balanced_df = balancer.balance_by_paper(train_df, 1)

    X_train, y_train = utils.make_feature_label_sets(train_balanced_df, new_features)
    X_test, y_test = utils.make_feature_label_sets(test_df, new_features)
    X_val, y_val = utils.make_feature_label_sets(val_df, new_features)

    classifier.fit(X_train, y_train)
    return ((y_true, classifier.predict(X_test)), (y_true_val, classifier.predict(X_val)))
