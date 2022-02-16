from libsvm.svmutil import svm_predict, svm_train
import time, sys
import numpy as np
from numpy.linalg import norm
import pandas as pd
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
from libsvm.svm import *
from libsvm.svmutil import *


def wTx(alpha, norms, SupportVectors, x):
    m = len(SupportVectors)
    temp = np.square(norms) + np.full((m, 1), norm(x) ** 2) - \
        2*np.dot(SupportVectors, x)
    temp = np.exp(-0.05*temp)
    return np.sum(alpha*temp)


def gaussian_kernal(X, Y, ind, c=1.0):
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
    sv_indices = [i for i in range(len(alpha_all)) if alpha_all[i] > 1.5*1e-3]
    alpha = [alpha_all[i]*Y[i] for i in sv_indices]
    SupportVectors = [X[i] for i in sv_indices]
    b = Y[ind]*1.0
    for i in sv_indices:
        b -= alpha_all[i] * Y[i] * K_temp[i][ind]
    return b, alpha_all, SupportVectors, sv_indices


def gaussian_predict(X, Y, b, alpha, SupportVectors):
    norms = np.array([norm(x) for x in SupportVectors]).reshape((-1, 1))
    prediction = [1 if wTx(alpha, norms, SupportVectors,
                           x.reshape((-1, 1))) + b >= 0 else -1 for x in X]
    accuracy = 100*sum([1 if Y[i] == prediction[i]
                       else 0 for i in range(len(Y))])/len(Y)
    return accuracy


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
    indices = [i for i in range(len(alpha)) if alpha[i] > 1e-4]
    alpha = [x for x in alpha if x > 1e-4]
    return w, b, alpha, SupportVectors, indices

def load_data(file, a, b):
    a, b = a % 10, b % 10
    data_all = pd.read_csv(file, header=None)
    data = data_all[(data_all[data_all.shape[1]-1] == a) |
                    (data_all[data_all.shape[1] - 1] == b)]
    y = np.array(data[data_all.shape[1] - 1])
    y = y.reshape(-1, 1)
    y[y == a] = 1
    y[y == b] = -1
    x = np.array(data.drop(data_all.shape[1] - 1, axis=1))
    x = x / 255.0
    return x, y


train_file = '/Users/aparahuja/Desktop/IITD/ML/Assignment 2/Q2/mnist/train.csv'
test_file = '/Users/aparahuja/Desktop/IITD/ML/Assignment 2/Q2/mnist/test.csv'

if len(sys.argv) > 1:
    train_file = sys.argv[1]
    test_file = sys.argv[2]

X, Y = load_data(train_file, 5, 6)
testX, testY = load_data(test_file, 5, 6)


print("LINEAR KERNAL\n")
st = time.time()
prob = svm_train(Y.reshape(-1), X, "-t 0 -q")
en = time.time()
print("LIBSVM Training Time = " + "{:.2f}".format(en - st) + " sec")

SupportVectorsIndices = prob.get_sv_indices()
alpha = prob.get_sv_coef()
w = np.zeros((X.shape[1], 1))
bias = 0
for i in range(len(SupportVectorsIndices)):
    index = SupportVectorsIndices[i] - 1
    w += alpha[i][0]*(X[index].reshape((-1, 1)))
for i in range(len(SupportVectorsIndices)):
    index = SupportVectorsIndices[i] - 1
    bias += Y[index] - np.dot(X[index], w)
b = 1.0*bias / len(SupportVectorsIndices)

st = time.time()
w_linear, b_linear, _, SupportVectors_linear, indices = linear_kernal(X, Y)
en = time.time()
print("CVXOPT Training Time = " + "{:.2f}".format(en - st) + " sec")

print("nSv for cvxopt implementation and libsvm:", len(indices), len(SupportVectorsIndices))
print("b for cvxopt implementation and libsvm:", b[0], b_linear)
print("max difference in weight for cvxopt implementation and libsvm:", max(np.absolute(w - w_linear))[0])

print("LIBSVM Training Data Accuracy -")
svm_predict(Y.reshape(-1), X, prob)

print("LIBSVM Testing Data Accuracy -")
svm_predict(testY.reshape(-1), testX, prob)


print("GAUSSIAN KERNAL\n")
st = time.time()
prob = svm_train(Y.reshape(-1), X, "-t 2 -g 0.05 -q")
en = time.time()
print("LIBSVM Training Time = " + "{:.2f}".format(en - st) + " sec")

SupportVectorsIndices = prob.get_sv_indices()
SupportVectorsIndices = [x-1 for x in SupportVectorsIndices]
alpha = prob.get_sv_coef()
alpha = [x[0] for x in alpha]
ind = -1
for i in range(len(alpha)):
    if abs(alpha[i]) < 0.9:
        ind = SupportVectorsIndices[i]
        break
b = Y[ind]*1.0
for i in range(len(alpha)):
    b -= alpha[i]*np.exp(-0.05*(norm(X[SupportVectorsIndices[i]] - X[ind])**2))

st = time.time()
b_gauss, alpha_all, SupportVectors, Indices = gaussian_kernal(X, Y, ind)
alpha_gauss = [alpha_all[i] * Y[i] for i in SupportVectorsIndices]
en = time.time()
print("CVXOPT Training Time = " + "{:.2f}".format(en - st) + " sec")

print("nSv for cvxopt implementation and libsvm:", len(Indices), len(SupportVectorsIndices))

sv1 = sorted(SupportVectorsIndices)
sv2 = sorted(Indices)
for i in range(len(sv1)):
    if sv1[i]!=sv2[i]:
        print("SVs don't match!")

print("b for cvxopt implementation and libsvm:", b_gauss[0], b[0])

max_diff = -1
for i in range(len(alpha)):
    max_diff = max(max_diff, abs(alpha[i] - alpha_gauss[i]))

# print(alpha, alpha_gauss)
print("max difference in alpha for cvxopt implementation and libsvm:", max_diff[0])

print("LIBSVM Training Data Accuracy -")
svm_predict(Y.reshape(-1), X, prob)

print("LIBSVM Testing Data Accuracy -")
svm_predict(testY.reshape(-1), testX, prob)
