#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import  classification_report, confusion_matrix
from sklearn import tree
import sys
from sklearn.metrics import recall_score
sys.path.append(os.path.abspath("../tools/"))
from feature_format import featureFormat, targetFeatureSplit
from sklearn.metrics import precision_score

data_dict = pd.read_pickle(r'C:\Users\imane\OneDrive\Desktop\Udacity_1\ud120-projects\final_project\final_project_dataset.pkl')

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
sort_keys = '../tools/python2_lesson14_keys.pkl'
labels, features = targetFeatureSplit(data)



### your code goes here 


X_train, X_test, y_train, y_test = train_test_split( features, labels, test_size=0.3, random_state=42)
dt = tree.DecisionTreeClassifier()
dt.fit(X_train, y_train)
pred = dt.predict(X_test)
poi_test = [i for i in y_test if i == 1]
print("POIs in the actual labels", len(poi_test))
poi_pred = [i for i in pred if i ==1]
print("POIs in the predictions ", len(poi_pred))
print(len(y_test))
print("Precision: ", precision_score(y_test, pred))

cm = confusion_matrix(y_test,pred)
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
        plt.show()

predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

cm = confusion_matrix(true_labels,predictions)
print(cm)
print("precision on hypothetical sets", precision_score(true_labels, predictions))
print("recall on hypothetical sets ",recall_score(true_labels,predictions))