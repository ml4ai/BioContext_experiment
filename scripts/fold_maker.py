"""
DESCRIPTION: This script is used to create a list of folds for running cross
             validation across multiple classifiers to ensure that the same
             validation sets are used for each test set across all different
             classifiers.

SCRIPT PARAMETERS: None

SCRIPT OUTPUT: Two pickle files holding CV folds for validation set size 3 and 4
"""
from os.path import join
import pickle

import utils.utils as utils
import utils.folds as folds


json_config = utils.Config()
DRIVE_PATH = json_config.get_features_filepath()
feature_path = join(DRIVE_PATH, "grouped_features.csv")
gdf = utils.load_dataframe(feature_path)

# Remove the degenerate paper
gdf2 = gdf[gdf.PMCID != "b'PMC4204162'"]

# Gather all paper-labels to make folds per-paper
row_labels = [pmc_id for pmc_id in gdf2["PMCID"].values]
paper_labels = list(set(row_labels))

print("Starting fold creation for val=4")
cv_folds_4 = folds.paper_fold_lists(paper_labels, row_labels)

print("Starting fold creation for val=3")
cv_folds_3 = folds.paper_fold_lists(paper_labels, row_labels, val_size=3)

print("Outputting fold sets")
pickle.dump(cv_folds_4, open(join(DRIVE_PATH, "cv_folds_val_4.pkl"), "wb"))
pickle.dump(cv_folds_3, open(join(DRIVE_PATH, "cv_folds_val_3.pkl"), "wb"))
