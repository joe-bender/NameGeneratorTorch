import pandas as pd

data = pd.read_csv('names2017.csv', header=None, index_col=False)
names = data.iloc[:, 0]
names = list(names)
names = [name for name in names if len(name) == 6]
print(len(names))
