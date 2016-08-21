# As with the previous exercises, let's look at the performance of a couple of classifiers
# on the familiar Titanic dataset. Add a train/test split, then store the results in the
# dictionary provided.

import numpy as np
import pandas as pd

# Load the dataset
X = pd.read_csv('titanic_data.csv')

X = X._get_numeric_data()
y = X['Survived']
del X['Age'], X['Survived']


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score as recall
from sklearn.metrics import precision_score as precision
from sklearn.naive_bayes import GaussianNB

# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
dt_recall = recall(clf.predict(X_test),y_test)
dt_prec = precision(clf.predict(X_test),y_test)
print "Decision Tree recall: {:.2f} and precision: {:.2f}".format(dt_recall, dt_prec)

clf = GaussianNB()
clf.fit(X_train, y_train)
nb_recall = recall(clf.predict(X_test),y_test)
nb_prec = precision(clf.predict(X_test), y_test)
print "GaussianNB recall: {:.2f} and precision: {:.2f}".format(nb_recall, nb_prec)

results = {
  "Naive Bayes Recall": nb_recall,
  "Naive Bayes Precision": nb_prec,
  "Decision Tree Recall": dt_recall,
  "Decision Tree Precision": dt_prec
}

print results