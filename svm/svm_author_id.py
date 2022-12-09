#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from time import time


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###
clf = SVC(kernel='rbf', C= 10000)

#########################################################

#########################################################
'''
You'll be Provided similar code in the Quiz
But the Code provided in Quiz has an Indexing issue
The Code Below solves that issue, So use this one
'''
# The training time takes too long, we can try splitting the training set into smaller samples
#features_train = features_train[:int(len(features_train)/100)]
#labels_train = labels_train[:int(len(labels_train)/100)]

#########################################################
t0 = time()
clf.fit(features_train,labels_train)
print("Training time: ", round(time()-t0, 3),"s")
t0 = time()
pred = clf.predict(features_test)
print("Testing time: ", round(time()-t0, 3), "s")
acc = clf.score(features_test,labels_test)
print("accuracy of svm is: ", acc)

p10 = pred[10]
p26 = pred[26]
p50 = pred[50]

print("email 10 ",p10)
print("email 26 ",p26)
print("email 50 ",p50)

chris_pred = []
for p in pred:
    if p == 1:
        chris_pred.append(p)
print("Number of Chris's predicted mails: ", len(chris_pred))

sara_pred = []
for p in pred:
    if p == 0:
        sara_pred.append(p)
print("Number of predicted sara mails: ", len(sara_pred))