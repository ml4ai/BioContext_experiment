import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import utils.utils as utils


json_config = utils.Config()
data_path = os.path.join(json_config.get_features_filepath(), "feature_comparison_results.csv")
classifier_data = dict()
feature_data = dict()
with open(data_path, "r+") as infile:
    for idx, line in enumerate(infile):
        if idx == 0:
            continue
        fields = line.split("\t")
        classifier = fields[0]
        f_set = fields[1]
        f1 = float(fields[-1])

        if classifier in classifier_data:
            classifier_data[classifier].append((f_set, f1))
        else:
            classifier_data[classifier] = [(f_set, f1)]

        if f_set not in feature_data:
            feature_data[f_set] = dict()
        feature_data[f_set][classifier] = f1


top_scores = [(name, max([score for _, score in data])) for name, data in classifier_data.items()]
classifier_names = [n for n, _ in sorted(top_scores, key=lambda tup: tup[1])]
classifier_idxs = [i for i in range(len(classifier_names))]

colors = ["#fc8d59", "#3288bd", "#e6f598", "#99d594", "#d53e4f"]
feature_set_names = ["CTX_EVT", "NDF", "NDF_CTX", "NDF_EVT", "All_features"]
f2c_map = {f: c for f, c in zip(feature_set_names, colors)}

plt.figure()
for idx, name in enumerate(classifier_names):
    data = classifier_data[name]
    (f_vals, y_vals) = map(list, zip(*data))
    x_vals = [idx for _ in y_vals]
    c_vals = [f2c_map[f] for f in f_vals]
    plt.scatter(x_vals, y_vals, c=c_vals, linewidths=1, edgecolors="black")

for f_set, classifier_data in feature_data.items():
    y_vals = [classifier_data[cl_name] for cl_name in classifier_names]
    c_idx = feature_set_names.index(f_set)
    plt.plot(classifier_idxs, y_vals, color=colors[c_idx], linewidth=1, linestyle="--")

plt.title("Comparative feature sets")
plt.xlabel("Classifier")
plt.ylabel("F1 score")
plt.xticks(classifier_idxs, classifier_names, rotation=15)
plt.yticks([i / 10 for i in range(1, 11)])
plt.grid(color='k', alpha=0.25, linestyle='--', linewidth=1)

proper_names = ["Ctx. DT and Evt. DT", "Non-DT Features", "Non DT and Ctx. DT", "Non DT and Evt. DT", "All Features"]
color_squares = [Rectangle((0, 1), 1, 1, color=c) for c in colors]
plt.legend(tuple(color_squares), tuple(proper_names))
plt.show()
