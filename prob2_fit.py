import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
CSCI 635: Introduction to Machine Learning
Problem 2: Polynomial Regression &

@author/lecturer - Alexander G. Ororbia II

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

# NOTE: you will need to tinker with the meta-parameters below yourself
#       (do not think of them as defaults by any means)
# meta-parameters for program
trial_name = 'p1_fit'  # will add a unique sub-string to output of this program
degree = 15  # p, order of model
beta = 1 # regularization coefficient
alpha = 0.04 # step size coefficient
eps = 0.0  # controls convergence criterion
n_epoch = 5000 # number of epochs (full passes through the dataset)


# begin simulation
def map_features(X, degree):
    index = 0
    Y = []
    while index < X.shape[0]:
        count = 1
        while count < degree + 1:
            temp = X[index][0] ** count
            Y.append(temp)
            count = count + 1
        index = index + 1
    temp = np.array(Y)
    # temp = temp.T
    Y = np.reshape(temp, (-1, degree))
    return Y


def regress(X, theta):
    ############################################################################
    # WRITEME: write your code here to complete the routine
    b, w = theta
    function = b + np.dot(X, w.T)
    return function


############################################################################

def gaussian_log_likelihood(mu, y):
    ############################################################################
    # WRITEME: write your code here to complete the routine
    cost = 0.0
    m = len(mu)
    index = 0
    while index < m:
        cost += (mu[index][0] - y[index][0]) ** 2
        index = index + 1
    return cost


def penalty(theta, beta, m):
    sum = np.sum(np.square(theta[1]))
    penalty = beta / (2 * m) * sum
    return penalty


############################################################################

def computeCost(X, y, theta, beta):  # loss is now Bernoulli cross-entropy/log likelihood
    ############################################################################
    # WRITEME: write your code here to complete the routine
    mu = regress(X, theta)
    cost = gaussian_log_likelihood(mu, y)
    m = len(mu)
    final_penalty = penalty(theta, beta, m)
    final_cost = final_penalty + cost
    return final_cost


############################################################################

def computeGrad(X, y, theta, beta):
    ############################################################################
    bias, weight = theta
    X = X.T
    y = y.T
    # HAD TO COMPUTE THIS LOCALLY TO FIT THE VECTOR, PROPERLY
    func_x = np.dot(weight,X) + bias
    size_x = len(X)
    size_w = len(weight[0])
    dL_db = np.average(func_x - y, axis=1)
    dL_dw = np.average(np.multiply(func_x - y, X), axis=1) + beta * np.average(weight, axis=1)
    return dL_db, dL_dw


############################################################################

path = os.getcwd() + '/data/prob2.dat'
data = pd.read_csv(path, header=None, names=['X', 'Y'])

# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:, 0:cols - 1]
y = data.iloc[:, cols - 1:cols]
data.info()
# convert from data frames to numpy matrices
X = np.array(X.values)
y = np.array(y.values)

############################################################################
# apply feature map to input features x1
# WRITEME: write code to turn X_feat into a polynomial feature map (hint: you
#          could use a loop and array concatenation)

############################################################################
X = map_features(X, degree)

# convert to numpy arrays and initialize the parameter array theta
w = np.zeros((1, X.shape[1]))
b = np.array([0])
theta = [b, w]
L = computeCost(X, y, theta, beta)
halt = 0  # halting variable (you can use these to terminate the loop if you have converged)
print("-1 L = {0}".format(L))
i = 0
while i < n_epoch and halt == 0:
    dL_db, dL_dw = computeGrad(X, y, theta, beta)
    b = theta[0]
    w = theta[1]

    b = b - alpha * dL_db
    w = w - alpha * dL_dw

    theta = b, w

    ############################################################################
    # update rules go here...
    # WRITEME: write your code here to perform a step of gradient descent & record anything else desired for later
    ############################################################################

    L = computeCost(X, y, theta, beta)

    ############################################################################
    # WRITEME: write code to perform a check for convergence (or simply to halt early)
    ############################################################################

    print(" {0} L = {1}".format(i, L))
    i += 1
# print parameter values found after the search
print("w = ", w)
print("b = ", b)

kludge = 0.25
# visualize the fit against the data
X_test = np.linspace(data.X.min(), data.X.max(), 100)
X_feat = np.expand_dims(X_test,
                        axis=1)  # we need this otherwise, the dimension is missing (turns shape(value, to shape(value,value))

############################################################################
# apply feature map to input features x1
# WRITEME: write code to turn X_feat into a polynomial feature map (hint: you
#          could use a loop and array concatenation)
############################################################################
X_feat = map_features(X_feat, degree)
plt.plot(X_test, regress(X_feat, theta), label="Model")
plt.scatter(X[:, 0], y, edgecolor='g', s=20, label="Samples")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim((np.amin(X_test) - kludge, np.amax(X_test) + kludge))
plt.ylim((np.amin(y) - kludge, np.amax(y) + kludge))
plt.legend(loc="best")

############################################################################
# WRITEME: write your code here to save plot to disk (look up documentation or
#          the inter-webs for matplotlib)
############################################################################
plt.savefig('out/prob2_fit_Q1Scatterplot0001.png')
plt.show()
