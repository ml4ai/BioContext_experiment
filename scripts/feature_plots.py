"""
DESCRIPTION: This script is used to see usage statistics of different features
             for each of the learning classifiers used in our experiments. The
             script finds the feature sets used in classification of each paper
             that recieved the highest testing F1 score. These per-paper feature
             uses are then aggregated per classifier to see how often each
             feature is used.

SCRIPT PARAMETERS: None

SCRIPT OUTPUT: Stacked bar plots show feature usage for each classifier
"""
import tqdm
import json
import os

import matplotlib.pyplot as plt
import numpy as np

import utils.utils as utils


# Load all ablation result files
json_config = utils.Config()
data_path = json_config.get_ablation_path()
files = [f for f in os.listdir(data_path) if f.endswith(".json")]
files = [
    "rbf_svm_ablation_results.json",
    "rand_forest_ablation_results.json",
    "neural_net_ablation_results.json",
    "grad_boost_ablation_results.json"
]
# Load data and get original feature names for display
frame_path = os.path.join(json_config.get_features_filepath(), "grouped_features.csv")
df = utils.load_dataframe(frame_path)
data_features = utils.data_features_only(df)
all_features = utils.get_original_features(data_features)

# Map features to a usable display name
feature_map = {
    "closesCtxOfClass": "Is ctx.\nclosest",
    "dependencyDistance": "Dependency\ndist.",
    "context_frequency": "Context\nfreq.",
    "sentenceDistance": "Sentence\ndist.",
    "evtSentencePresentTense": "Evt sent.\nPresent\ntense",
    "evtSentencePastTense": "Evt. sent.\nPast tense",
    "ctxSentencePresentTense": "Ctx. sent.\nPresent\ntense",
    "ctxSentencePastTense": "Ctx. sent.\nPast tense",
    "evtNegationInTail": "Negated\nevent\nmention",
    "ctxNegationIntTail": "Negated\ncontext\nmention",
    "CTX_DEP_TAIL_FEATS": "Ctx.\nspanning\ndep.\nbigrams",
    "EVT_DEP_TAIL_FEATS": "Evt.\nspanning\ndep.\nbigrams",
    "ctxSentenceFirstPerson": "Ctx, sent.\nin first\nperson",
    "evtSentenceFirstPerson": "Evt, sent.\nin first\nperson",
}

feature_names = [feature_map[feat] for feat in all_features]

all_model_features = dict()
for filename in files:
    model_name = utils.resolve_model_name(filename)
    feature_data = json.load(open(os.path.join(data_path, filename)))

    # Extract score and best feature set per-paper for the current classifier
    paper_scores = {pid: [] for pid in feature_data[0]["by_paper_validation"].keys()}
    for feat_dict in tqdm.tqdm(feature_data, desc="Processing {}".format(model_name)):
        for pid, scores in feat_dict["by_paper_validation"].items():
            f1 = utils.f1(scores)
            paper_scores[pid].append((f1, feat_dict["features"]))

    def best_score(scores):
        (f1, features) = max(scores, key=lambda tup: tup[0])
        return utils.get_original_features(features)

    # Grab the features from the best feature set for each paper
    used_features = {pid: best_score(scores) for pid, scores in paper_scores.items()}
    all_model_features[model_name] = used_features

# Count feature occurrences over each paper for each classifier
per_model_occurrences = dict()
for mod_name, paper_feats in all_model_features.items():
    feature_occurrences = {feat: 0 for feat in all_features}
    for pid, features in paper_feats.items():
        for feat in features:
            feature_occurrences[feat] += 1
    per_model_occurrences[mod_name] = feature_occurrences

# Plot the results in a stacked bar-plot that shares the x-axis
x_vals = np.array(list(range(len(all_features))))
f, axarr = plt.subplots(4, sharex=True)
for i, (mod_name, feature_occurrences) in enumerate(per_model_occurrences.items()):
    y_occs = [feature_occurrences[feat] for feat in all_features]
    axarr[i].bar(x_vals, y_occs)
    axarr[i].set_xlabel("{}".format(mod_name))
    axarr[i].set_ylim(0, 22)    # Ensure all y-axis ranges are [0 - num papers]

    if i == 0:
        axarr[i].set_title("Frequency of feature uses in classification")

plt.xticks(x_vals, feature_names)
plt.show()
