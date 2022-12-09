#!/usr/bin/python3
import os
import joblib
import sys
import matplotlib.pyplot
sys.path.append(os.path.abspath("../tools/"))
from feature_format import featureFormat, targetFeatureSplit
import pandas as pd
import numpy as np


### read in data dictionary, convert to numpy array
data_dict = joblib.load( open("../final_project/final_project_dataset.pkl", "rb") )
features = ["salary", "bonus"]
# delete the outlier (which is the row with total values of columns
del data_dict["TOTAL"]
data = featureFormat(data_dict, features)
target, features = targetFeatureSplit(data)

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()



# Find outliers

df = pd.DataFrame.from_dict(data_dict)
df = df.T
df = df.reset_index()

n_df = pd.DataFrame(df, columns=["index", "salary", "bonus"])
# get the index of the row with the highest salary
#print(n_df[n_df["salary"]== 26704229 ].index.values)
# get the name of that data point
#print(n_df.iloc[[103]])

# get the index of the two people who made over 5 million dollars in bonuses and over 1 million dollars in salary

n_df["salary"] = n_df["salary"].astype(float)
n_df["bonus"] = n_df["bonus"].astype(float)
print(n_df[n_df["salary"] > 1000000].index.values)
print(n_df.iloc[[64]])
print(n_df.iloc[[95]])
print(n_df.iloc[[127]])