"""
DESCRIPTION: This script outputs the P/R/F1 scores for all learning classifiers
             as well as the baseline scores. The scores are output in a bar plot
             with labeled heights and bolded max values for P/R/F1. To run this
             script you must have ablation results for each classifier you wish
             to view all in the same directory

SCRIPT PARAMETERS: None

SCRIPT OUTPUT: Plot showing a comparsion of P/R/F1 scores across all classifiers
"""
import tqdm
import json
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from utils.classifiers import baseline_counts
import utils.utils as utils


# Load all ablation result files
json_config = utils.Config()
data_path = json_config.get_ablation_path()
feat_path = json_config.get_features_filepath()
files = [f for f in os.listdir(data_path) if ".json" in f]

overall_scores = list()
for filename in files:
    model_name = utils.resolve_model_name(filename)
    feature_data = json.load(open(os.path.join(data_path, filename)))

    # Extract score and best feature set per-paper for the current classifier
    paper_scores = {pid: [] for pid in feature_data[0]["by_paper_validation"].keys()}
    for feat_dict in tqdm.tqdm(feature_data, desc="Processing {}".format(model_name)):
        for pid, scores in feat_dict["by_paper_validation"].items():
            f1 = utils.f1(scores)
            paper_scores[pid].append((f1, scores,
                                      feat_dict["by_paper_test"][pid],
                                      feat_dict["features"]))

    def best_score(scores):
        (f1, val_preds, test_preds, _) = max(scores, key=lambda tup: tup[0])
        return (f1, val_preds, test_preds)

    # Extract best val/test scores and associate with paper-id
    best_paper_scores = {pid: best_score(scores) for pid, scores in paper_scores.items()}

    # Sum T/F P/N counts across all papers
    test_counts = {
        "TP": sum([scores["TP"] for (_, _, scores) in best_paper_scores.values()]),
        "FP": sum([scores["FP"] for (_, _, scores) in best_paper_scores.values()]),
        "TN": sum([scores["TN"] for (_, _, scores) in best_paper_scores.values()]),
        "FN": sum([scores["FN"] for (_, _, scores) in best_paper_scores.values()])
    }

    # Score and record P/R/F1 for the classifier
    test_f1 = utils.f1(test_counts)
    test_prec = utils.precision(test_counts)
    test_recall = utils.recall(test_counts)
    overall_scores.append((model_name, test_prec, test_recall, test_f1))

    print("Results for {}".format(model_name))
    print("Precision: {:.4f}\tRecall: {:.4f}\tF1: {:.4f}".format(test_prec, test_recall, test_f1))

# Score and record baseline values
print("Computing baseline values")
baseline_scores = baseline_counts(feat_path)
base_f1 = utils.f1(baseline_scores)
base_prec = utils.precision(baseline_scores)
base_recall = utils.recall(baseline_scores)
overall_scores.append(("Baseline", base_prec, base_recall, base_f1))
print("Baseline F1 score: {}".format(base_f1))

overall_scores.sort(key=lambda tup: tup[3])
(names, p_scores, r_scores, f1_scores) = map(list, zip(*overall_scores))
x_vals = np.array(list(range(len(names))))

# Write out all scores to a csv file
with open("../data/classifier_scores.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Model name", "Precision", "Recall", "F1"])
    writer.writerows(overall_scores)

# Plot the scores in a bar-plot
plt.figure(figsize=(10, 6))
rects1 = plt.bar(x_vals - 0.2, p_scores, width=0.2, color="#fc8d59", align='center', label="Precision")
rects2 = plt.bar(x_vals, r_scores, width=0.2, color="#99d594", align='center', label="Recall")
rects3 = plt.bar(x_vals + 0.2, f1_scores, width=0.2, color="#2b83ba", align='center', label="F1")

# Include a horizontal line at the level of the Baseline F1 score
plt.hlines(base_f1, x_vals[0]-0.3, x_vals[-1]+0.3, linestyles="dashed", colors="red")
plt.xticks(x_vals, names, rotation=15)
plt.ylim((0, 1))


def autolabel(rects, scores):
    """
    Attach a text label above each bar displaying its height
    """
    max_score = max(scores)
    for rect, score in zip(rects, scores):
        height = rect.get_height()
        weight = "bold" if score == max_score else "normal"
        plt.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                 "{:.3f}".format(score),
                 ha='center', va='bottom', weight=weight)


# Label the height of all bars with actual scores
autolabel(rects1, p_scores)
autolabel(rects2, r_scores)
autolabel(rects3, f1_scores)

plt.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.17))
plt.title("Classifier score comparison")
plt.xlabel("Model name")
plt.ylabel("%-score")

plt.subplots_adjust(left=0.075, right=.925, bottom=.2, top=.95)

plt.show()
