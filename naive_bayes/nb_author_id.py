#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
from sklearn.naive_bayes import GaussianNB
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


# Import the model
NB = GaussianNB()


'''
You Will be Required to record time for Training and Predicting 
The Code Given on Udacity Website is in Python-2
The Following Code is Python-3 version of the same code
'''
# Train the model and keep track of the time taken
t0 = time()
NB.fit(features_train,labels_train)
print("Training Time:", round(time()-t0, 3), "s")

# Make predictions and keep track of the time taken
t0 = time()
pred = NB.predict(features_test)
print("Predicting Time:", round(time()-t0, 3), "s")

##############################################################
# Get accuracy
accuracy = NB.score(features_test,labels_test)
print("Accuracy is: ", accuracy)
