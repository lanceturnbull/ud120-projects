# In this exercise we'll examine a learner which has high variance, and tries to learn
# nonexistant patterns in the data.
# Use the learning curve function from sklearn.learning_curve to plot learning curves
# of both training and testing error. Use plt.plot() within the plot_curve function
# to create line graphs of the values.

from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve
from sklearn.cross_validation import KFold
from sklearn.metrics import explained_variance_score, make_scorer
import numpy as np

size = 600000
cv = KFold(size,shuffle=True)
score = make_scorer(explained_variance_score)

X = np.round(np.reshape(np.random.normal(scale=5,size=2*size),(-1,2)),2)
y = np.array([[np.sin(x[0]+np.sin(x[1]))] for x in X])

def plot_curve():
    # YOUR CODE HERE
    reg = DecisionTreeRegressor()
    reg.fit(X,y)
    print reg.score(X,y)

    # TODO: Create the learning curve with the cv and score parameters defined above.
    train_size, train_score, test_score = learning_curve(reg, X, y, cv=cv)

    # TODO: Plot the training and testing curves.
    plt.plot(train_size, np.mean(train_score, axis=1), color='g', label='training score')
    plt.plot(train_size, np.mean(test_score, axis=1), color='r', label='test score')
    plt.legend(loc="best")

    # Show the result, scaling the axis for visibility
    plt.ylim(-.1,1.1)
    plt.show()


plot_curve()