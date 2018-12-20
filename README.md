# Bio-Context: Experimental Code Repository
This repository servers as a container for all of the scripts and utility code required to replicate the results found and published as part of our paper that was accepted to the DMBIH '18 workshop. The following sections denote all of the prerequisites required to run our code and the functionality of our scripts that are all included under the `scripts/` directory. This repository has a sister repositories named `BioContext_results` that houses large JSON files that include our per-classifier feature set results that are used to generate the results shown in our paper. Please consider cloning and using these results if you do not have access to the HPC resources necessary to replicate the exhaustive search we used to find the optimal feature sets.

### Prerequisites
- Unzip the file located at `data/features.csv.zip`; this is the features data that is needed to run nearly all of our scripts used in this repository
- Set the two absolute paths required in `config.json`
- All scripts require a minimum of `Python 3.5.2` however `Python 3.6.1` is preferred
- Run `pip -r requirements.txt` to install all of the required python dependencies.
- Add the following path to your `PYTHONPATH` environment variable: `/Absolute/path/to/BioContext_experiment/`

### Replication Instructions
1. Run the script `balance_and_group.py` in order to get `grouped_features.csv` that will be used for all future scripts
2. Run the script `fold_maker.py` to obtain fold sets that can be used during the ablation study for the various classifiers
3. Using a large distributed/parallel resource run the script `ablation_study.py` with the command shown above. Run this once for each classifier you with to verify.
4. Use the ablation results with our remaining scripts to get the results you require from our paper. The scripts are commented with a description of the results they will provide.

### Script Specifics
- `ablation_study.py` -- Used to run the massive distributed parallel search over all feature sets for a particular classifier. When supplied with the name of a classifier, this will output a large JSON file with per-paper TP/FP/TN/FN counts, for every possible feature set, from our cross-validation study.
  - Run this script with the following command: `mpirun -n <number-of-cores> python ablation_study.py <classifier-identifier>`
  - Acceptable classifier identifiers are:
    - `log_reg` -- to use the Logistic Regression classifier
    - `linear_svm` -- to use the SVM classifier with a linear kernel
    - `poly_svm` -- to use the SVM classifier with a polynomial kernel
    - `rbf_svm` -- to use the SVM classifier with a gaussian kernel
    - `neural_net` -- to use the Feed-forward NN classifier
    - `rand_forest` -- to use the Random Forest classifier


- `balance_and_group.py` -- This script handles the class imbalance issue in `features.csv` that results from the annotation of only positive context instances. This script also handles grouping the data into Event-Context pairs and goes further to group the data by PMC-paper-ID in order to assist in the creation of per-paper folds for cross-validation.

- `classification_results.py` -- Searches the ablation data obtained from the `BioContext_results` repository to find and report the P/R/F1 scores for each classifier aggregated over papers.

- `feature_plots.py` -- This script creates stacked bar plots that represent counts of features that are used in the classification of papers during cross-validation. One bar plot is made for each classifier and each bar plot has an entry for each feature that is used for each paper.

- `fold_maker.py` -- Used to make (train, validation, test) folds that can be used across all the different classifiers so that we have folds that were randomly generated, and yet are the same for each classifier.

- `significance.py` -- This script runs bootstrap statistical significance sampling when provided with the name of a specific JSON file containing ablation data for a particular classifier (i.e. from our `BioContext_results` repository) and an increase in F1 score over the baseline to test (0 is a good value to use as a starting point).

- `throwout.py` -- This script is used to show the P/R/F1 scores needed to verify that the one paper we chose to exclude from our results is indeed an outlier that has extremely poor performance when compared with the other papers.
