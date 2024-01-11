import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# meta-parameters for program
alpha = 0.001  # step size coefficient
eps = 0.1  # controls convergence criterion
n_epoch = 40  # number of epochs (full passes through the dataset)


# begin simulation


def regress(X, theta):
    ############################################################################
    b,w = theta
    return X * w + b


############################################################################
def linear_function(X, theta):
    return np.dot(theta[1].T, X.T) + theta[0]


############################################################################

def gaussian_log_likelihood(mu, y):
    ############################################################################
    MSE = np.average(np.square(np.subtract(mu, y.T)))
    MSE = MSE / 2
    return MSE


############################################################################

def computeCost(X, y, theta):  # loss:  Bernoulli cross-entropy/log likelihood
    mu = linear_function(X, theta)
    cost = gaussian_log_likelihood(mu, y)
    print(cost)
    return cost


############################################################################

def computeGrad(X, y, theta):
    ############################################################################

    dL_dfy = None  # derivative w.r.t. to model output units (fy)
    dL_db = None  # derivative w.r.t. model weights w
    dL_dw = None  # derivative w.r.t model bias b
    func_x = linear_function(X, theta)
    dL_db = np.subtract(func_x, y).mean()
    dL_dw = np.multiply((np.subtract(func_x, y)), X).mean()
    nabla = (dL_db, dL_dw)  # nabla represents the full gradient
    return nabla


############################################################################

path = os.getcwd() + '/data/prob1.dat'
data = pd.read_csv(path, header=None, names=['X', 'Y'])

# print some stats about the data
data.info()

# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:, 0:cols - 1]
y = data.iloc[:, cols - 1:cols]

# convert from data frames to numpy matrices
X = np.array(X.values)
y = np.array(y.values)
e = plt.figure(1)
plt.scatter(X, y, c='pink')
# convert to numpy arrays and initialize the parameter array theta
w = np.zeros((1, X.shape[1]))
b = np.array([0])
theta = (b, w)

L = computeCost(X, y, theta)
print("-1 L = {0}".format(L))
L_best = L
halt = 0  # halting variable
i = 0
cost = []  # keep track of our loss function
while i < n_epoch and halt == 0:
    dL_db, dL_dw = computeGrad(X, y, theta)
    b = theta[0]
    w = theta[1]
    b[0] = b[0] - dL_db * alpha
    w[0][0] = w[0][0] - dL_dw * alpha


    L = computeCost(X, y, theta)  # track our loss after performing a single step
    cost.append(L)

    print(" {0} L = {1}".format(i, L))
    i += 1
# print parameter values found after the search
print("w = ", w)
print("b = ", b)

kludge = 0.25  # helps with printing the plots
# visualize the fit against the data
X_test = np.linspace(data.X.min(), data.X.max(), 100)
X_test = np.expand_dims(X_test, axis=1)

f = plt.figure(2)

plt.plot(X_test, regress(X_test, theta), label="Model")
plt.scatter(X[:, 0], y, edgecolor='g', s=20, label="Samples")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim((np.amin(X_test) - kludge, np.amax(X_test) + kludge))
plt.ylim((np.amin(y) - kludge, np.amax(y) + kludge))
plt.legend(loc="best")
f.savefig('out/prob1_fit_Q1Scatterplot.png')
g = plt.figure(3)
iteration = list(range(1, i + 1))
plt.plot(iteration, cost, label='cost v/s epoch')
plt.xlabel("iterations")
plt.ylabel("cost")
g.savefig('out/prob1_fit_Q1ivsc.png')

plt.show()
