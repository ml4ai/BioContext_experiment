from multiprocessing import Pool, cpu_count

from tqdm import tqdm
import numpy as np
import pandas


def perform_grouping(data_frame, features, grouping_key):
    """
    This function creates a dataframe whose rows represent groups of rows in
    the original dataframe. The list of features to be includes in the new
    dataframe is specified by the grouping key. The columns in the new
    dataframe will include three columns per feature (one for min, max, and avg)
    as well as a column for the PMCID, the label and all columns used in the
    grouping keyself.

    :data_frame:    A pandas DataFrame object with the initial data rows
    :features:      A list of features to include in the new DataFrame
    :grouping_key:  A tuple used to create groups of rows from data_frame

    :returns:       A pandas DataFrame where rows represent groups of rows in
                    the original data_frame
    """
    grouped_data = list(data_frame.groupby(grouping_key))
    input_data = [(group, grouping_key, features) for group in grouped_data]

    p = Pool(cpu_count())
    res = list(tqdm(p.imap(group_rows, input_data), total=len(grouped_data)))

    # Merging individual results from multicore grouping op
    data_dict = res[0]
    for res_dict in res[1:]:
        for key in data_dict.keys():
            data_dict[key].extend(res_dict[key])

    # Extending feature list for new DataFrame
    new_features = list()
    for feature in features:
        name_min = "{}_min".format(feature)
        name_max = "{}_max".format(feature)
        name_avg = "{}_avg".format(feature)
        new_features.extend([name_min, name_avg, name_max])

    # Creating new DataFrame object with grouped data
    new_df = pandas.DataFrame.from_dict(data_dict)
    return (new_df, new_features)


def calculate_statistics(frame, feature):
    """
    Returns the min, max, and average values of a feature in a dataframe.

    :frame: A pandas dataframe that contains a given feature, feature
    :feature: A string key whose entries are to be used to calculate min, max, avg

    :returns: (Int, Int, Int) --> the min, max, avg of the feature values found
    """
    values = frame[feature]
    return np.min(values), np.max(values), np.mean(values)


def group_rows(data):
    """
    Given a set of rows that all represent the same Evt-Ctx pair, we group the
    data values of the rows by taking the min, max, and average of the each
    column. These values then create a row triple the size of the original row
    for each Evt-Ctx pair containing information about the min, max, and average
    of the feature values from the original rows that described an Evt-Ctx pair.
    We have found that this method prevents data loss in our system during
    classification.

    :data: a tuple that contains all information needed to describe a group of
           Evt-Ctx pairs
    :return: a data dictionary object that holds a row for each Evt-Ctx pair of
             the extended row type described above
    """
    ((idx, frame), grouping_key, features) = data

    # Adding entries to data dictionary for every member of the grouping key,
    # the PMC ID of the paper, and the label. These fields must always be saved
    data_dict = {key: list() for key in grouping_key}
    data_dict["label"] = list()
    if "PMCID" not in grouping_key:
        data_dict["PMCID"] = list()

    # Adding entries to data dictionary for every feature we are to include
    new_features = list()
    for feature in features:
        name_min = "{}_min".format(feature)
        name_max = "{}_max".format(feature)
        name_avg = "{}_avg".format(feature)
        data_dict[name_min] = list()
        data_dict[name_avg] = list()
        data_dict[name_max] = list()
        new_features.extend([name_min, name_avg, name_max])

    # Add the label for this group to the DataFrame
    data_dict["label"].append(any(frame['label']))

    # Add PMCID to new DataFrame if not already in the grouping key
    if "PMCID" not in grouping_key:
        PMCID_list = list(frame['PMCID'])
        data_dict["PMCID"].append(PMCID_list[0])

    # Add elements from grouping key to new DataFrame
    for i, key in enumerate(grouping_key):
        data_dict[key].append(idx[i])

    # min,max,avg for each feature in the given group
    for f in features:
        min_val, max_val, avg_val = calculate_statistics(frame, f)
        data_dict["{}_min".format(f)].append(min_val)
        data_dict["{}_max".format(f)].append(max_val)
        data_dict["{}_avg".format(f)].append(avg_val)

    return data_dict
