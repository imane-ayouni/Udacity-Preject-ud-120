#!/usr/bin/python3

import joblib
import numpy
numpy.random.seed(42)
import pandas as pd


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = r"C:\Users\imane\OneDrive\Desktop\Udacity_1\ud120-projects\feature_selection\your_word_data.pkl"
authors_file = r"C:\Users\imane\OneDrive\Desktop\Udacity_1\ud120-projects\feature_selection\you_email_authors.pkl"
word_data = pd.read_pickle(words_file)
authors = pd.read_pickle(authors_file)



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')

features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()



names = vectorizer.get_feature_names_out()

### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]



### your code goes here

from sklearn import tree
from sklearn.metrics import accuracy_score

clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)

clf.predict(features_test)
print("acc on testing set: ",clf.score(features_test,labels_test))

importances = clf.feature_importances_
for index, item in enumerate(importances):
    if item > 0.2:
       print(index, item)

