# import pandas
import os
import matplotlib.pyplot as plt

import utils.utils as utils


json_config = utils.Config()
feat_data_path = os.path.join(json_config.get_features_filepath(), "features.csv")
print("loading data")
df = utils.load_dataframe(feat_data_path)
grouping_key = tuple(["EvtID", "CtxID"])
print("Grouping data")
grouped_data = list(df.groupby(grouping_key))
pair_sizes = [frame.shape[0] for (idx, frame) in grouped_data]
print(pair_sizes)
print(grouped_data[0])
print(grouped_data[1])
plt.figure()
plt.hist(pair_sizes, bins="auto")
plt.show()
