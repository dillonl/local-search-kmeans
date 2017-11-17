import pandas as pd
import numpy as np

data = pd.read_csv('data/data.csv')
data = data.drop(data.columns[[0]], axis=1)
print(data.describe())
