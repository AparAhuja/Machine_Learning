import numpy as np
import sys
from numpy.linalg import norm
import pandas as pd
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

k = 10
train_file = '/Users/aparahuja/Desktop/IITD/ML/Assignment 2/Q2/mnist/train.csv'
test_file = '/Users/aparahuja/Desktop/IITD/ML/Assignment 2/Q2/mnist/test.csv'

if len(sys.argv) > 1:
    train_file = sys.argv[1]
    test_file = sys.argv[2]


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


def load_data_all(file):
    data = pd.read_csv(file, header=None)
    y = np.array(data[data.shape[1] - 1])
    y = y.reshape(-1, 1)
    x = np.array(data.drop(data.shape[1] - 1, axis=1))
    x = x / 255.0
    return x, y


def F1_Confusion(y_true, y_predict):
    confusion = np.zeros((10, 10))
    n = len(y_true)
    for i in range(n):
        confusion[int(y_true[i])][int(y_predict[i])] += 1
    print("Confusion Matrix: ")
    print(confusion)
    maxDiagonal = max([i for i in range(10)], key=lambda x: confusion[x][x])
    print("Max Diagonal Entry in confusion matrix occurs for the class - " + str(maxDiagonal))
    print("F1 scores:")
    f1_avg = 0
    for i in range(10):
        tp = confusion[i][i]
        fn = sum([confusion[j][i] if i != j else 0 for j in range(10)])
        fp = sum([confusion[i][j] if i != j else 0 for j in range(10)])
        f1_score = tp/(tp+(fp+fn)/2)
        f1_avg += f1_score/10
        print("\tClass " + str(i) + " = " + "{:.5f}".format(f1_score))
    print("Macro F1 score = " + "{:.5f}".format(f1_avg))


def wTx(alpha, norms, SupportVectors, X):
    m = len(SupportVectors)
    alpha = np.array(alpha).reshape((-1, 1))
    SupportVectors = np.array(SupportVectors).reshape((m, -1))
    K = -2.0*np.dot(SupportVectors, X.T)
    for i in range(m):
        for j in range(m):
            K[i][j] += norms[i]**2 + norm(X[j])**2
    K = np.exp(-0.05*K)
    return np.sum(alpha*K, axis=0)


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
        threshold = min(alpha_all)
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


def gaussian_predict(X, b, alpha, SupportVectors):
    norms = [norm(x) for x in SupportVectors]
    return wTx(alpha, norms, SupportVectors, X) + b[0]


def multclass_predict(X, Y):
    print("Starting Testing -")
    correct = 0
    total = len(X)
    y_true = []
    y_pred = []
    cnt = [[0 for i in range(10)] for j in range(total)]
    score = [[0 for i in range(10)] for j in range(total)]
    t = 0
    for j in range(k):
        for l in range(k):
            if j > l:
                t += 1
                # print(t)
                X_train, Y_train = load_data(train_file, j, l)
                b, alpha, SupportVectors = gaussian_kernal(X_train, Y_train)
                pred_jl = gaussian_predict(X, b, alpha, SupportVectors)
                for i in range(total):
                    if pred_jl[i] >= 0:
                        cnt[i][j] += 1
                        score[i][j] = max(score[i][j], abs(pred_jl[i]))
                    else:
                        cnt[i][l] += 1
                        score[i][l] = max(score[i][l], abs(pred_jl[i]))
    y_true = [x[0] for x in Y]
    for i in range(total):
        v = max(cnt[i])
        indices = []
        for j in range(k):
            if cnt[i][j] == v:
                indices.append(j)
        max_score = -1
        ans = -1
        for x in indices:
            if score[i][x] > max_score:
                max_score = max(max_score, score[i][x])
                ans = x
        y_pred.append(ans)
        correct += (1 if y_pred[i] == y_true[i] else 0)
    F1_Confusion(y_true, y_pred)
    print("Accuracy = " + "{:.2f}".format(100*correct/total) + "%")


testX, testY = load_data_all(test_file)
multclass_predict(testX, testY)
