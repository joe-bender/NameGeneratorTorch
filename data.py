import torch
import pandas as pd
import helpers

def get_names(filename):
    # we don't nead any header or index, just the names list
    data = pd.read_csv(filename, header=None, index_col=False)
    # only keep the first column, which has the names
    names = data.iloc[:, 0]
    # turn the pandas column into a python list
    names = list(names)
    return names

print(len(get_names('names2017.csv')))
