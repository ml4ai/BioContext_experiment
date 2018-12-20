import numpy as np
import pandas


def random_balancer(df, amt_per_pos):
    """
    :df:    A pandas dataframe object containing all row data
    :amt_per_pos:   The amount of negative examples to accept per positive example

    :return: A pandas dataframe with a balanced amount of true/false context instances
    """
    chosen_rows = random_row_selection(df, amt_per_pos)
    new_df = pandas.DataFrame(chosen_rows, columns=df.columns)
    return new_df


def random_row_selection(df, amt_per_pos):
    """
    Creates a numpy array of rows from all rows in a dataframe such that there
    is a balanced number of positive and negative context instances. Negative
    instances are selected at random so that there are amt_per_pos negative
    context instances for every positive context instance.

    :df:    A pandas dataframe containing all row data to choose from
    :amt_per_pos:   A integer number of neg instances for each pos instance
    :returns:   A numpy array of chosen rows
    """
    # Get all rows in DataFrame with a true/False label value
    pos_df = df[df["label"] == True]
    neg_df = df[df["label"] == False]

    # Count numbers of Pos/Neg instances in DataFrame
    num_neg = neg_df.shape[0]
    num_pos = pos_df.shape[0]

    # Balance the Pos/Neg classes by choosing randomly from the class with more
    # values to match the size of the class with fewer values
    if num_neg < num_pos:
        amt_pos_examples = num_neg * amt_per_pos
        if amt_pos_examples > num_pos:
            raise ValueError("Requested balancing requires more pos examples than total present.")

        all_pos_rows = pos_df.values

        # Shuffle allows for a grabbing selection to be random
        np.random.shuffle(all_pos_rows)
        chosen_pos_rows = all_pos_rows[: amt_pos_examples]

        # Create one large set of rows from label partitions
        all_rows = np.concatenate((neg_df.values, chosen_pos_rows), axis=0)
    else:
        amt_neg_examples = num_pos * amt_per_pos
        if amt_neg_examples > num_neg:
            raise ValueError("Requested balancing requires more neg examples than total present.")

        all_neg_rows = neg_df.values

        # Shuffle allows for a grabbing selection to be random
        np.random.shuffle(all_neg_rows)
        chosen_neg_rows = all_neg_rows[: amt_neg_examples]

        # Create one large set of rows from label partitions
        all_rows = np.concatenate((pos_df.values, chosen_neg_rows), axis=0)

    return all_rows


def balance_by_paper(df, amt_per_pos):
    """
    This function should take in a dataframe and return a dataframe with an
    equivalent number of positive and negative context instances per PMCID.
    Negative instances for a paper should still be chosen at random from the
    set of all negative instances for the paper.

    NOTE: it could be the case that a paper may have more negative instances
          than positive instances. This needs to be verified and if more
          positive instances are present than negative then the balancing must
          be reversed.

    :df: The initial dataframe with all row data
    :amt_per_pos: Integer amount of negative examples to include per positive

    :return: A new dataframe, still sorted by paper, with balanced pos/neg instances
    """

    """
    Note: a possible approach might be to group the dataframe by PMCID,
    and then invoke the random_balancer on each group.
    Since at any time we will have a finite number of papers, the loop should run in linear time.
    """
    # Random seed ensures that balancing chooses the same random values for
    # CV-folds w/ validation set size 3 or 4 (For val-set-size experiment)
    np.random.seed(13)
    grouping_key = "PMCID"
    grouped_by_paper = df.groupby(grouping_key)
    grouped_by_paper = list(grouped_by_paper)

    # Create per-paper balanced sets of rows
    all_paper_rows = None
    for idx, paper_df in grouped_by_paper:
        if all_paper_rows is None:
            all_paper_rows = random_row_selection(paper_df, amt_per_pos)
        else:
            chosen_rows = random_row_selection(paper_df, amt_per_pos)
            all_paper_rows = np.concatenate((all_paper_rows, chosen_rows), axis=0)

    return pandas.DataFrame(all_paper_rows, columns=df.columns)
