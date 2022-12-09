#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""
import os
import joblib
import sys
sys.path.append(os.path.abspath("../tools/"))
from feature_format import featureFormat, targetFeatureSplit
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
data_dict = pd.read_pickle(r'C:\Users\imane\OneDrive\Desktop\Udacity_1\ud120-projects\final_project\final_project_dataset.pkl')
#data_dict = joblib.load(open("../final_project/final_project_dataset.pkl", "rb") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
sort_keys = '../tools/python2_lesson13_keys.pkl'
labels, features = targetFeatureSplit(data)

X_train, X_test, y_train, y_test = train_test_split( features, labels, test_size=0.2, random_state=42)

### it's all yours from here forward!  


dt = tree.DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt.predict(X_test)
print(dt.score(X_test, y_test))