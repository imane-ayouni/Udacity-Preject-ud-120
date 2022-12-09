#!/usr/bin/python

import sys
import pickle
import os
sys.path.append(os.path.abspath(("../tools/")))
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn import tree
from xgboost import XGBClassifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred',
                 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other',
                 'long_term_incentive', 'restricted_stock', 'director_fees'] # You will need to use more features


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# At first I will use all the features and determine as I go which ones are important so I can keep them

random_forest = RandomForestClassifier()
random_forest.fit(features, labels)
ft_imp = random_forest.feature_importances_
print(ft_imp)
# features importance determined that 5 features have more than 0.091 importance
# I picked that threshold because it seems reasonable,
features_list = ["poi",'total_payments','bonus','expenses', 'exercised_stock_options',  'other' ]
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 2: Remove outliers
# Thanks to some pre processing work I've done where I turned the dataset to a pandas dataframe
# and with the help of boxplots determined the threshold above which a point is an outlier
# I will replace the outliers with the average of the feature

# first let's calculate averages
payments = []
for i in range(len(features)):
    payments.append(features[i][0])
a_payments = (sum(payments)/len(payments))
bonus = []
for i in range(len(features)):
    bonus.append((features[i][1]))
a_bonus = sum(bonus)/len(bonus)
expenses = []
for i in range(len(features)):
    expenses.append(features[i][2])
a_expenses = sum(expenses)/len(expenses)
stock = []
for i in range(len(features)):
    stock.append(features[i][3])
a_stock = sum(stock)/len(stock)
other = []
for i in range(len(features)):
    other.append(features[i][4])
a_other = sum(other)/len(other)


for i in range(len(features)):
    if features[i][0] >= 100000000:
        features[i][0] = a_payments
    else:
        pass

for i in range(len(features)):
    if features[i][1] >= 80000000:
        features[i][1] = a_bonus
    else:
        pass
for i in range(len(features)):
    if features[i][2] >= 5000000:
        features[i][2] = a_expenses
    else:
        pass
for i in range(len(features)):
    if features[i][3] >= 300000000:
        features[i][3] = a_stock
    else:
        pass
for i in range(len(features)):
    if features[i][4] >= 40000000:
        features[i][4] = a_other
    else:
        pass

# Before creating new features with the use of PCA, I need to scale the data
# I'm going to use the min max scaler for this one

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
scaler = MinMaxScaler()
scaler.fit(features_train)
features_train = scaler.transform(features_train)
features_test = scaler.transform(features_test)

### Task 3: Create new feature(s)

pca_5 = PCA(n_components=5)
pca_5.fit(features_train)
print("eigenvalues for pca with 5 components",pca_5.explained_variance_ratio_)
# by looking at the eigenvalue and data variance, I determined that the first three components are the ones
# that hold most of the information, so I'm gonna make a pca with only 3 components
pca_3 = PCA(n_components=3)
pca_3.fit(features_train)
print("eigenvalues for pca with 3 components",pca_3.explained_variance_ratio_)
features_train = pca_3.transform(features_train)
features_test = pca_3.transform(features_test)




### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# next is to perform a grid search to determine best parameters
params_NB = {'var_smoothing': np.logspace(0,-9, num=100)}
gs_NB = GridSearchCV(estimator=clf,
                 param_grid=params_NB,
                 cv= 3,   # use any cross validation technique
                 verbose=1,
                 scoring='accuracy')
gs_NB.fit(features_train, labels_train)

print("best params NB", gs_NB.best_params_)
# best var_smoothing is set to 0.432
clf = GaussianNB(var_smoothing = 0.43287612810830584)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

print(classification_report(labels_test, pred))
print("accuracy NB", clf.score(features_test, labels_test))
print("precision NB", precision_score(labels_test,pred ))
print("recall NB", recall_score(labels_test, pred))

# Since the performance of Gaussian NB is very bad when it comes to identifying the people of interest,
# I will test other models
clf = tree.DecisionTreeClassifier(random_state=42)
param_grid = {'max_features': ['auto', 'sqrt', 'log2'],
              'ccp_alpha': [0.1, .01, .001],
              'max_depth' : [5, 6, 7, 8, 9],
              'criterion' :['gini', 'entropy']}
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, verbose=True)
grid_search.fit(features_train, labels_train)
print("best params DT", grid_search.best_estimator_)
clf = tree.DecisionTreeClassifier(ccp_alpha=0.01, max_depth=5, max_features='auto',
                       random_state=42)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

print(classification_report(labels_test, pred))
print("accuracy DT", clf.score(features_test, labels_test))
print("precision DT", precision_score(labels_test,pred ))
print("recall DT", recall_score(labels_test, pred))

# The performance of decision tree isn't better than that of Gaussian NB, the accuracy has dropped while the precision and recall are still at 0
# Let's try XGB classifier
clf = XGBClassifier(
    objective= 'binary:logistic',
    nthread=4,
    seed=42
)
parameters = {
    'max_depth': range (2, 10, 1),
    'n_estimators': range(60, 220, 40),
    'learning_rate': [0.1, 0.01, 0.05]
}
grid_search = GridSearchCV(
    estimator=clf,
    param_grid=parameters,
    scoring = 'roc_auc',
    n_jobs = 10,
    cv = 3,
    verbose=True
)
grid_search.fit(features_train, labels_train)
print("best params XGB", grid_search.best_estimator_)
clf = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
              gamma=0, gpu_id=-1, importance_type=None,
              interaction_constraints='', learning_rate=0.05, max_delta_step=0,
              max_depth=4, min_child_weight=1,
              monotone_constraints='()', n_estimators=100, n_jobs=4, nthread=4,
              num_parallel_tree=1, predictor='auto', random_state=42,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=42,
              subsample=1, tree_method='exact', validate_parameters=1,
              verbosity=None)

clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

print(classification_report(labels_test, pred))
print("accuracy xgb", clf.score(features_test, labels_test))
print("precision xgb", precision_score(labels_test,pred ))
print("recall xgb", recall_score(labels_test, pred))

# The accuracy , precision and recall of xgb is similar to that of Gaussian NB

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)





































