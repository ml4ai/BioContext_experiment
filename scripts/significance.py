"""
DESCRIPTION: This script runs bootstrap significance testing of our
             classification results. Run this script with increasing values of
             <points gained in F1> (Range: [0, 1]) until the output is at the
             desired confidence level.

SCRIPT PARAMETERS: name of result file (JSON), <points gained in F1>

SCRIPT OUTPUT: The confidence in the classifier from the loaded file reaching the
               <points gained in F1> over the baseline classifier
"""
import json
import sys
import os

from sklearn import metrics
from tqdm import tqdm
import numpy as np

from utils.classifiers import baseline_counts
import utils.utils as utils


def main():
    if len(sys.argv) != 3:
        raise RuntimeError("Script must be run with two arguments: <JSON-darta-file> and <F1-score-increase>")
        sys.exit(1)

    classifier_json = sys.argv[1]
    amt_above_baseline = float(sys.argv[2])

    json_config = utils.Config()
    feat_path = json_config.get_features_filepath()

    # Our baseline F1 score for comparison to our classifier F1 score
    base_f1 = utils.f1(baseline_counts(feat_path))

    data_path = json_config.get_ablation_path()
    json_path = os.path.join(data_path, classifier_json)
    feature_data = json.load(open(json_path))

    # A list of the form (prediction, label) from the best feature set (by
    # validation score) for this classifier. NOTE that the predictions are the
    # test predictions for this classifier from the best performing feature set
    # during the validation phase. Sample from these prediction/label pairs to
    # conduct our bootstrapping significance test
    preds_and_labels = reconstruct_pred_label_list(feature_data)

    # Use indexes of (label, prediction) pairs to work with numpy.random.choice()
    indexOfTruth = [i for i in range(len(preds_and_labels))]
    num_iterations = 1000
    counts = list()
    for k in tqdm(range(num_iterations), desc="Scoring samples"):
        # Select samples WITH REPLACEMENT from preds_and_labels
        indices_of_samples = np.random.choice(indexOfTruth, size=len(preds_and_labels), replace=True)
        samples = [preds_and_labels[idx] for idx in indices_of_samples]
        (preds, labels) = zip(*samples)

        # Compute the F1 score of the selected sample.
        f1_score = metrics.f1_score(labels, preds)

        # Test to see if this F1 score is X points greater than the baseline F1 score
        is_above_baseline = f1_score - amt_above_baseline > base_f1
        counts.append(1 if is_above_baseline else 0)

    # Sum the recorded scores and divide by the size of the loop and report result
    result = sum(counts) / num_iterations
    print("We are {:.2f}%% confident that our model increases F1".format(result * 100) +
          " performance by {} points over the baseline.".format(amt_above_baseline))


def reconstruct_pred_label_list(json_data):
    # Extract best scores from best feature set on a per-paper basis
    paper_scores = {pid: [] for pid in json_data[0]["by_paper_validation"].keys()}
    for feat_dict in tqdm(json_data, desc="Processing feature sets"):
        for pid, scores in feat_dict["by_paper_validation"].items():
            f1 = utils.f1(scores)
            paper_scores[pid].append((f1, feat_dict["by_paper_test"][pid]))

    # Collect the test scores from the best-scoring feature set for this paper
    def best_score(scores):
        (f1, test_preds) = max(scores, key=lambda tup: tup[0])
        return test_preds

    # Asscoiate best scoring counts with their respective paper-id
    per_paper_counts = {pid: best_score(scores) for pid, scores in paper_scores.items()}

    # Create sum of T/F P/N counts
    best_counts = {
        "TP": sum([scores["TP"] for scores in per_paper_counts.values()]),
        "FP": sum([scores["FP"] for scores in per_paper_counts.values()]),
        "TN": sum([scores["TN"] for scores in per_paper_counts.values()]),
        "FN": sum([scores["FN"] for scores in per_paper_counts.values()])
    }

    # Reconstruct Label/Predicted value pairs from T/F P/N counts
    return [(True, True) for i in range(best_counts["TP"])] + \
        [(True, False) for i in range(best_counts["FP"])] + \
        [(False, False) for i in range(best_counts["TN"])] + \
        [(False, True) for i in range(best_counts["FN"])]


if __name__ == '__main__':
    main()
