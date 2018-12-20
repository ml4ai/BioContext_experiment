"""
DESCRIPTION: This script creates a new DataFrame from our initial DataFrame
             stored in features.csv. The reason for making this new DataFrame is
             two-fold. First we need to transform our intial data that has
             multiple entries per Event-Context pair into a single row entry per
             Event-Context pair. After that we wish to correct for class
             imbalancing amongst the label field. To do this we will randomly
             select rows from whichever label has more values to create an equal
             number of T/F labels. Once this has been done the final DataFrame
             will be saved to a file named grouped_features.csv in the same
             location as features.csv

SCRIPT PARAMETERS: None

SCRIPT OUTPUT: New saved DataFrame named grouped_features.csv
"""
import utils.balancer as balancer
import utils.grouper as grouper
import utils.utils as utils

json_cofig = utils.Config()
data_path = json_cofig.get_features_filepath()

df = utils.load_dataframe(data_path + "features.csv")

print("Balancing Positive/Negative context classes")
pre_balanced_df = balancer.balance_by_paper(df, 1)
print("Number of rows in balanced df: {}".format(len(pre_balanced_df)))

print("Grouping rows by (PMCID, EvtID, CtxID) key pairs")
grouping_key = ["PMCID", "EvtID", "CtxID"]
data_features = utils.data_features_only(df)
res = grouper.perform_grouping(pre_balanced_df, data_features, grouping_key)
(dataframe_by_groups, new_features) = res

print("Outputting grouped dataframe")
dataframe_by_groups.to_csv(data_path + "grouped_features.csv")
