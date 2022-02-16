# My entry number is 2019CS10465.
import time
import sys
from typing import SupportsComplex
from nltk.corpus.reader.chasen import test
import numpy as np
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


def linear_kernal(X, Y, c=1.0):
    # =  alphai * yi * xi.T * yj * xj * alphaj
    P = cvxopt_matrix(np.dot(Y*X, (Y*X).T), tc='d')
    # = -1*alphai
    q = cvxopt_matrix(np.full((X.shape[0], 1), -1), tc='d')
    # -1*alphai <= 0 and alphai <= c
    G = cvxopt_matrix(
        np.vstack((-1*np.eye(X.shape[0]), np.eye(X.shape[0]))), tc='d')
    h = cvxopt_matrix(
        np.vstack((np.zeros((X.shape[0], 1)), c*np.ones((X.shape[0], 1)))), tc='d')
    A = cvxopt_matrix(Y.reshape(1, -1), tc='d')
    b = cvxopt_matrix(np.array([0]), tc='d')
    cvxopt_solvers.options['show_progress'] = False
    cvxopt_solver = cvxopt_solvers.qp(P, q, G, h, A, b)
    alpha = np.array(cvxopt_solver['x'])
    Support = (alpha > 1e-4)
    Support = Support.flatten()
    w = np.dot((Y[Support] * alpha[Support]).T, X[Support])
    w = w.reshape(-1, 1)
    b = np.mean(Y[Support] - np.dot(X[Support], w))
    SupportVectors = [X[i] for i in range(len(alpha)) if alpha[i] > 1e-4]
    alpha = [x for x in alpha if x > 1e-4]
    return w, b, alpha, SupportVectors


def linear_predict(X, Y, w, b):
    prediction = [1.0 if np.dot(x.reshape((1, -1)), w)[0][0] + b >= 0 else -1.0 for x in X]
    accuracy = 100*sum([(1 if Y[i][0] == prediction[i]
                       else 0) for i in range(len(Y))])/len(Y)
    return accuracy


train_file = '/Users/aparahuja/Desktop/IITD/ML/Assignment 2/Q2/mnist/train.csv'
test_file = '/Users/aparahuja/Desktop/IITD/ML/Assignment 2/Q2/mnist/test.csv'

if len(sys.argv) > 1:
    train_file = sys.argv[1]
    test_file = sys.argv[2]

X, Y = load_data(train_file, 5, 6)
testX, testY = load_data(test_file, 5, 6)

st = time.time()
w, b, alpha, SupportVectors = linear_kernal(X, Y)
en = time.time()
print("Training Time = " + "{:.2f}".format(en - st) + " sec")
print(len(SupportVectors))
print("Training Accuracy = " +
      "{:.2f}".format(linear_predict(X, Y, w, b)) + "%")
print("Testing Accuracy = " +
      "{:.2f}".format(linear_predict(testX, testY, w, b)) + "%")
