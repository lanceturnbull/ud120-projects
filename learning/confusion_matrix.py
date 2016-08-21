# In this exercise, we'll use the Titanic dataset as before, train two classifiers and
# look at their confusion matrices. Your job is to create a train/test split in the data
# and report the results in the dictionary at the bottom.

import numpy as np
import pandas as pd

# Load the dataset
from sklearn import datasets

X = pd.read_csv('titanic_data.csv')

X = X._get_numeric_data()
y = X['Survived']
del X['Age'], X['Survived']


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB

# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
print "Confusion matrix for this Decision Tree:\n",confusion_matrix(clf.predict(X_test),y_test)

clf = GaussianNB()
clf.fit(X_train, y_train)
print "GaussianNB confusion matrix:\n",confusion_matrix(clf.predict(X_test), y_test)

#TODO: store the confusion matrices on the test sets below

confusions = {
 "Naive Bayes": 0,
 "Decision Tree": 0
}