

from sklearn.naive_bayes import GaussianNB
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData
from time import time
#def classify(features_train, labels_train):
#    clf = GaussianNB()
#    clf.fit(features_train, labels_train)

X_train, Y_train, X_test, Y_test = makeTerrainData
clf = GaussianNB()
t0 = time()
clf.fit(X_train,Y_train)
print("Train time ",round(time()-t0,3),"s")
t0 =time()
print("Test time ",round(time()-t0,3),"s")
pred = clf.predict(X_test)



### import the sklearn module for GaussianNB
### create classifier
### fit the classifier on the training features and labels
### return the fit classifier


### your code goes here!




