#!/usr/bin/python

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
sys.path.append(os.path.abspath(("../tools/")))
from sklearn.preprocessing import MinMaxScaler
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV



### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary',
                  'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred',
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

# create a dataframe to store the information and make its manipulation easier
l = []
for i in my_dataset.items():
    l.append(i)
names = []
for i in l:
    name = i[0]
    names.append(name)
columns = []
for i in l:
    col = i[1]
    columns.append(col)
df = pd.DataFrame(columns)
df["names"] = names
poi = df['poi']

# let's seperate the dataset into financial data and email data
df_financial = df.copy()
for i in df_financial:
    if i in features_list:
        pass
    else:
        del df_financial[i]
df_financial["names"] = names
df_financial = df_financial.set_index("names")

df_emails = df.copy()
for i in df_emails:
    if i in features_list:
        del df_emails[i]
    else:
        pass


# Check for null values
sns.heatmap(df_financial.isnull())
plt.show()

# replace NaN with 0s

df_financial = df_financial.astype(np.float)


# filter important features usine

random_forest = RandomForestClassifier()
random_forest.fit(features, labels)
ft_imp = random_forest.feature_importances_
print(ft_imp)
# From the feature importance I will pick 5 features that are most important and correlated with the target
important = ['total_payments','bonus','expenses', 'exercised_stock_options',  'other' ]
for i in df_financial:
    if i in important:
        pass
    else:
        del df_financial[i]


### Task 2: Remove outliers
# First of all, let's visualize outliers with the help of box plots
df_financial.boxplot(column="total_payments")
plt.show()
df_financial.boxplot(column="bonus")
plt.show()
df_financial.boxplot(column="expenses")
plt.show()
df_financial.boxplot(column="exercised_stock_options")
plt.show()
df_financial.boxplot(column="other")
plt.show()

# With the help of the boxplots, I have determined thresholds above which the point is an outlier,
# I will go ahead and replace the outliers with the means
mean_payments = df_financial["total_payments"].mean()
for i in df_financial["total_payments"]:
    if i >= 100000000:
        df_financial['total_payments'] = df_financial['total_payments'].replace(i,mean_payments)
    else:
        pass
mean_bonus = df_financial["bonus"].mean()
for i in df_financial["bonus"]:
    if i >= 80000000:
        df_financial['bonus'] = df_financial['bonus'].replace(i, mean_bonus)
    else:
        pass
mean_expenses = df_financial["expenses"].mean()
for i in df_financial["expenses"]:
    if i >= 5000000:
        df_financial['expenses'] = df_financial['expenses'].replace(i, mean_expenses)
    else:
        pass

mean_stock = df_financial["exercised_stock_options"].mean()
for i in df_financial["exercised_stock_options"]:
    if i >= 300000000:
        df_financial['exercised_stock_options'] = df_financial['exercised_stock_options'].replace(i, mean_stock)
    else:
        pass
mean_other = df_financial["other"].mean()
for i in df_financial["other"]:
    if i >= 40000000:
        df_financial['other'] = df_financial['other'].replace(i, mean_other)
    else:
        pass

# Replacing NaN with the mean of the column
mean_payments = df_financial["total_payments"].mean()
for i in df_financial["total_payments"]:
    if i == "nan":
        df_financial['total_payments'] = df_financial['total_payments'].replace(i,mean_payments)
    else:
        pass
mean_bonus = df_financial["bonus"].mean()
for i in df_financial["bonus"]:
    if i == "nan":
        df_financial['bonus'] = df_financial['bonus'].replace(i, mean_bonus)
    else:
        pass
mean_expenses = df_financial["expenses"].mean()
for i in df_financial["expenses"]:
    if i == "nan":
        df_financial['expenses'] = df_financial['expenses'].replace(i, mean_expenses)
    else:
        pass

mean_stock = df_financial["exercised_stock_options"].mean()
for i in df_financial["exercised_stock_options"]:
    if i == "nan":
        df_financial['exercised_stock_options'] = df_financial['exercised_stock_options'].replace(i, mean_stock)
    else:
        pass
mean_other = df_financial["other"].mean()
for i in df_financial["other"]:
    if i == "nan":
        df_financial['other'] = df_financial['other'].replace(i, mean_other)
    else:
        pass



### Task 3: Create new feature(s)
# In this part I will just add some features from the emails dataset that I think will help me with the classification
# The features that I'm going to add are the number of messages from/to poi and the person in question
# and whether the poi has been tagged in this email
df_financial["from_poi_to_this_person"] = df_emails["from_poi_to_this_person"]
df_financial["from_this_person_to_poi"] = df_emails["from_this_person_to_poi"]
df_financial["shared_receipt_with_poi"] = df_emails["shared_receipt_with_poi"]

# Replacing nan with the 0

for i in df_financial["from_poi_to_this_person"]:
    if i == "NaN":
        df_financial['from_poi_to_this_person'] = df_financial['from_poi_to_this_person'].replace(i, 0)
    else:
        pass

for i in df_financial["from_this_person_to_poi"]:
    if i == "NaN":
        df_financial['from_this_person_to_poi'] = df_financial['from_this_person_to_poi'].replace(i, 0)
    else:
        pass

for i in df_financial["shared_receipt_with_poi"]:
    if i == "NaN":
        df_financial['shared_receipt_with_poi'] = df_financial['shared_receipt_with_poi'].replace(i, 0)
    else:
        pass

for i in df["poi"]:
    if i == "True":
        df['poi'] = df['poi'].replace(i, 1)
    else:
        df['poi'] = df['poi'].replace(i, 0)

# Seperating the target
target = df["poi"]


# Seperating training and testing sets
features_train, features_test, labels_train, labels_test = train_test_split(df_financial, target, test_size=0.3, random_state=42)
# Scaling the data
scaler = MinMaxScaler()
scaler.fit(features_train)
features_train = scaler.transform(features_train)
features_test = scaler.transform(features_test)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
# next is to perform a grid search to determine best parameters
params_NB = {'var_smoothing': np.logspace(0,-9, num=100)}
gs_NB = GridSearchCV(estimator=clf,
                 param_grid=params_NB,
                 cv= 3,   # use any cross validation technique
                 verbose=1,
                 scoring='accuracy')
gs_NB.fit(features_train, labels_train)

print("best params", gs_NB.best_params_)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)




