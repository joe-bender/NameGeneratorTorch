"""Collect data from a text file to be used by the network"""

import pandas as pd

def get_names(filename):
    """Get a list of all the names from the given text file

    * filename: the file to get the names from
    """

    # we don't nead any header or index, just the names list
    data = pd.read_csv(filename, header=None, index_col=False)
    # only keep the first column, which has the names
    names = data.iloc[:, 0]
    # turn the pandas column into a python list
    names = list(names)

    return names
