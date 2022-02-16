import numpy as np
import time
import sys
from numpy.linalg import norm
import pandas as pd
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

def load_data(file, a, b):
    a, b = a % 10, b % 10
    data_all = pd.read_csv(file, header=None)
    data = data_all[(data_all[data_all.shape[1] - 1] == a) |
                    (data_all[data_all.shape[1] - 1] == b)]
    y = np.array(data[data_all.shape[1] - 1])
    y = y.reshape(-1, 1)
    y[y == a] = 1
    y[y == b] = -1
    x = np.array(data.drop(data_all.shape[1] - 1, axis=1))
    x = x / 255.0
    return x, y


def wTx(alpha, norms, SupportVectors, x):
    m = len(SupportVectors)
    if m ==0:
        return 0
    temp = np.square(norms) + np.full((m, 1), norm(x)** 2) - 2*np.dot(SupportVectors, x)
    temp = np.exp(-0.05*temp)
    return np.sum(alpha*temp)


def gaussian_kernal(X, Y, c=1.0):
    m = X.shape[0]
    K = -2.0*np.dot(X, X.T)
    norms = [norm(x) for x in X]
    for i in range(m):
        for j in range(m):
            K[i][j] += norms[i]**2 + norms[j]**2
    K_temp = np.exp(-0.05*K)
    K = K_temp * np.dot(Y, Y.T)
    # =  alphai * yi * yj * <xi, xj> * alphaj
    P = cvxopt_matrix(K, tc='d')
    # = -1*alphai
    q = cvxopt_matrix(np.full((X.shape[0], 1), -1), tc='d')
    # -1*alphai <=0 and alphai <= c
    G = cvxopt_matrix(
        np.vstack((-1*np.eye(X.shape[0]), np.eye(X.shape[0]))), tc='d')
    h = cvxopt_matrix(
        np.vstack((np.zeros((X.shape[0], 1)), c*np.ones((X.shape[0], 1)))), tc='d')
    A = cvxopt_matrix(Y.reshape(1, -1), tc='d')
    b = cvxopt_matrix(np.array([0]), tc='d')
    cvxopt_solvers.options['show_progress'] = False
    cvxopt_solver = cvxopt_solvers.qp(P, q, G, h, A, b)
    alpha_all = np.array(cvxopt_solver['x'])
    threshold = 1.5*1e-3
    if len([i for i in range(len(alpha_all)) if alpha_all[i] > threshold]) == 0:
        threshold = 1e-4
    if len([i for i in range(len(alpha_all)) if alpha_all[i] > threshold]) == 0:
        threshold = 1e-5
    if len([i for i in range(len(alpha_all)) if alpha_all[i] > threshold]) == 0:
        threshold = 1e-6
    sv_indices = [i for i in range(len(alpha_all)) if alpha_all[i] >= threshold]
    alpha = [alpha_all[i]*Y[i] for i in sv_indices]
    SupportVectors = [X[i] for i in sv_indices]
    ind = -1
    for i in sv_indices:
        if alpha_all[i] < 0.9:
            ind = i
            break
    b = Y[ind]*1.0
    for i in sv_indices:
        b -= alpha_all[i] * Y[i] * K_temp[i][ind]
    return b, alpha, SupportVectors


def gaussian_predict(X, Y, b, alpha, SupportVectors):
    norms = np.array([norm(x) for x in SupportVectors]).reshape((-1, 1))
    prediction = [1 if wTx(alpha, norms, SupportVectors, x.reshape((-1, 1))) + b >= 0 else -1 for x in X]
    accuracy = 100*sum([1 if Y[i] == prediction[i] else 0 for i in range(len(Y))])/len(Y)
    return accuracy


train_file = '/Users/aparahuja/Desktop/IITD/ML/Assignment 2/Q2/mnist/train.csv'
test_file = '/Users/aparahuja/Desktop/IITD/ML/Assignment 2/Q2/mnist/test.csv'

if len(sys.argv) > 1:
    train_file = sys.argv[1]
    test_file = sys.argv[2]

X, Y = load_data(train_file, 5, 6)
testX, testY = load_data(test_file, 5, 6)
st = time.time()
b, alpha, SupportVectors = gaussian_kernal(X, Y)
en = time.time()
print("Training Time = " + "{:.2f}".format(en - st) + " sec")
print("Training Accuracy = " +
      "{:.2f}".format(gaussian_predict(X, Y, b, alpha, SupportVectors)) + "%")
print("Testing Accuracy = " +
      "{:.2f}".format(gaussian_predict(testX, testY, b, alpha, SupportVectors)) + "%")
