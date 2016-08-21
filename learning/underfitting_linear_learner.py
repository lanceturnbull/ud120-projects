# In this exercise we'll examine a learner which has high bias, and is incapable of
# learning the patterns in the data.
# Use the learning curve function from sklearn.learning_curve to plot learning curves
# of both training and testing error. Use plt.plot() within the plot_curve function
# to create line graphs of the values.

from sklearn.linear_model import LinearRegression
from sklearn.learning_curve import learning_curve
import matplotlib.pyplot as plt
from sklearn.metrics import explained_variance_score, make_scorer
from sklearn.cross_validation import KFold
import numpy as np

size = 1000
cv = KFold(size, shuffle=True)
score = make_scorer(explained_variance_score)

X = np.reshape(np.random.normal(scale=2, size=size), (-1, 1))
y = np.array([[1 - 2 * x[0] + x[0] ** 2] for x in X])


def plot_curve():
    reg = LinearRegression()
    reg.fit(X, y)
    print reg.score(X, y)

    # TODO: Create the learning curve with the cv and score parameters defined above.
    train_num, train_score, test_score = learning_curve(reg, X, y, cv=cv)
    plt.title("accuracy")
    plt.xlabel("Training size")
    plt.ylabel('Score')
    plt.grid()
    # TODO: Plot the training and testing curves.
    plt.plot(train_num, np.mean(train_score, axis=1), 'o-', color="r", label='Train score')
    plt.plot(train_num, np.mean(test_score, axis=1), 'o-', color="g", label="Test score")
    plt.legend(loc="best")
    # Sizes the window for readability and displays the plot.
    plt.ylim(-.1, 1.1)
    plt.show()

plot_curve()