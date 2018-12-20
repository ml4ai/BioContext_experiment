"""
DESCRIPTION: This script creates a plot to show per-paper F1 scores of all
             the classifiers. The papers on the x-axis are sorted via the
             baseline F1 score for the paper. The baseline score is represented
             by a red 'x', the top scoring classifier overall is represented
             by a black 'x'. All other classifiers are color-coded via a legend.

SCRIPT PARAMETERS: None

SCRIPT OUTPUT: Scatter-plot of per-paper, per-classifier F1 scores
"""
import tqdm
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from utils.classifiers import per_fold_baseline_counts
import utils.utils as utils


json_config = utils.Config()
data_path = json_config.get_ablation_path()
# files = [f for f in os.listdir(data_path) if ".json" in f]
files = [
    "rbf_svm_ablation_results.json",
    "rand_forest_ablation_results.json",
    "neural_net_ablation_results.json",
    "grad_boost_ablation_results.json"
]

model_scores = dict()
paper_labels = None
paper_counts = None

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
        (f1, val_preds, test_preds, feats) = max(scores, key=lambda tup: tup[0])
        best_features = utils.get_original_features(feats)
        return (f1, val_preds, test_preds, best_features)

    # Gather best T/F P/N counts along with the feature set used to generate them
    # and associate these with the paper that produced the data used in classification
    best_paper_scores = {pid: best_score(scores) for pid, scores in paper_scores.items()}
    paper_labels = list(best_paper_scores.keys())

    # Get a total number of predictions made per paper (only need this once)
    if paper_counts is None:
        paper_counts = {pid: sum([preds for preds in pred_dict.values()])
                        for pid, (_, pred_dict, _, _) in best_paper_scores.items()}

    # add scores for this classifier to overall model scores
    model_scores[model_name] = dict()
    for paper_id, (_, _, test_preds, _) in best_paper_scores.items():
        model_scores[model_name][paper_id] = utils.f1(test_preds)

print("Computing baseline values")
baseline_scores = per_fold_baseline_counts()
model_scores["Baseline"] = baseline_scores

baseline_tups = sorted(list(baseline_scores.items()), key=lambda tup: tup[1])
(paper_labels, _) = map(list, zip(*baseline_tups))

colors = ["black", "#fc8d59", "#3288bd", "#99d594", "#d53e4f"]  # "#e6f598",
x_vals = np.array(list(range(len(paper_labels))))
plt.figure(figsize=(14, 8))

model_names = sorted(list(model_scores.keys()))
for idx, name in enumerate(model_names):
    model_dict = model_scores[name]
    # labels = list(model_dict.keys())
    # x_locs = [paper_labels.index(label) for label in labels]
    x_locs = list(range(len(paper_labels)))
    scores = [model_dict[label] for label in paper_labels]

    # Plot the per paper scores for this model (one scatter point per paper)
    m = "x" if name == "Baseline" else "o"
    plt.scatter(x_locs, scores, color=colors[idx], label=name, marker=m, linewidths=1, edgecolors="black")
    plt.plot(x_locs, scores, color=colors[idx], linestyle="--", linewidth=1)

# Create the paper display labels with total # of predictions in the paper
disp_labels = ["#{}\n({})".format(p[-4: -1], paper_counts[p]) for p in paper_labels]
plt.xticks(x_vals, disp_labels)
plt.ylim((0, 1))
plt.legend(ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.17))
plt.title("Per-paper F1 score analysis")
plt.xlabel("Paper")
plt.ylabel("%-score")

plt.grid(color='k', alpha=0.25, linestyle='--', linewidth=1)

plt.subplots_adjust(left=0.075, right=.925, bottom=.2, top=.95)
plt.show()
