import json
import sys
import os

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import utils.utils as utils


def main():
    json_config = utils.Config()
    feat_path = json_config.get_features_filepath()
    pred_path = os.path.join(feat_path, "per-paper-preds/")

    num_iterations = 10000
    c1_name = sys.argv[1]
    c2_name = sys.argv[2]
    c3_name = sys.argv[3]
    c4_name = sys.argv[4]

    c1_print_name = utils.get_classifier_print_name(c1_name)
    c2_print_name = utils.get_classifier_print_name(c2_name)
    c3_print_name = utils.get_classifier_print_name(c3_name)
    c4_print_name = utils.get_classifier_print_name(c4_name)

    c1_preds = json.load(open(os.path.join(pred_path, "{}.json".format(c1_name))))
    c2_preds = json.load(open(os.path.join(pred_path, "{}.json".format(c2_name))))
    c3_preds = json.load(open(os.path.join(pred_path, "{}.json".format(c3_name))))
    c4_preds = json.load(open(os.path.join(pred_path, "{}.json".format(c4_name))))

    papers = sorted(list(c1_preds.keys()))

    diffs_1_2 = list()
    for i in tqdm(range(num_iterations), desc="Run bootstrapping"):
        sample = np.random.choice(papers, size=len(papers), replace=True)
        c1_f1_score = score_sampled_preds(sample, c1_preds)
        c2_f1_score = score_sampled_preds(sample, c2_preds)
        diffs_1_2.append(c1_f1_score - c2_f1_score)

    (values, bins_1_2) = np.histogram(diffs_1_2, bins="auto", density=True)
    bin_width_1_2 = bins_1_2[1] - bins_1_2[0]
    values_1_2 = [val * (bin_width_1_2) for val in values]

    diffs_1_3 = list()
    for i in tqdm(range(num_iterations), desc="Run bootstrapping"):
        sample = np.random.choice(papers, size=len(papers), replace=True)
        c1_f1_score = score_sampled_preds(sample, c1_preds)
        c3_f1_score = score_sampled_preds(sample, c3_preds)
        diffs_1_3.append(c1_f1_score - c3_f1_score)

    (values, bins_1_3) = np.histogram(diffs_1_3, bins="auto", density=True)
    bin_width_1_3 = bins_1_3[1] - bins_1_3[0]
    values_1_3 = [val * (bin_width_1_3) for val in values]

    diffs_1_4 = list()
    for i in tqdm(range(num_iterations), desc="Run bootstrapping"):
        sample = np.random.choice(papers, size=len(papers), replace=True)
        c1_f1_score = score_sampled_preds(sample, c1_preds)
        c4_f1_score = score_sampled_preds(sample, c4_preds)
        diffs_1_4.append(c1_f1_score - c4_f1_score)

    (values, bins_1_4) = np.histogram(diffs_1_4, bins="auto", density=True)
    bin_width_1_4 = bins_1_4[1] - bins_1_4[0]
    values_1_4 = [val * (bin_width_1_4) for val in values]

    # plt.figure()
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    # plt.title("{} as compared to other classifiers".format(c1_print_name))
    # plt.xlabel("F1 score difference")
    # plt.ylabel("Percentage of Predictions")

    ax1.set_title("({} - {})".format(c1_print_name, c2_print_name))
    ax1.bar(bins_1_2[:-1], values_1_2, width=bin_width_1_2)
    # ax1.set_xlabel("F1 score difference")
    ax1.set_ylabel("Percentage of Predictions")

    ax2.set_title("({} - {})".format(c1_print_name, c3_print_name))
    ax2.bar(bins_1_3[:-1], values_1_3, width=bin_width_1_3)
    ax2.set_xlabel("F1 score difference")
    # ax2.set_ylabel("Percentage of Predictions")

    ax3.set_title("({} - {})".format(c1_print_name, c4_print_name))
    ax3.bar(bins_1_4[:-1], values_1_4, width=bin_width_1_4)
    # ax3.set_xlabel("F1 score difference")
    # ax3.set_ylabel("Percentage of Predictions")

    plt.show()


def score_sampled_preds(sample, score_dict):
    sampled_preds = {pid: score_dict[pid] for pid in sample}

    return utils.f1({
        "TP": sum([scores["TP"] for scores in sampled_preds.values()]),
        "FP": sum([scores["FP"] for scores in sampled_preds.values()]),
        "TN": sum([scores["TN"] for scores in sampled_preds.values()]),
        "FN": sum([scores["FN"] for scores in sampled_preds.values()])
    })


if __name__ == '__main__':
    main()
