from libsvm.svmutil import svm_predict, svm_train
import matplotlib.pyplot as plt
import numpy as np, sys
import pandas as pd
from libsvm.svm import *
from libsvm.svmutil import *

train_file = '/Users/aparahuja/Desktop/IITD/ML/Assignment 2/Q2/mnist/train.csv'
test_file = '/Users/aparahuja/Desktop/IITD/ML/Assignment 2/Q2/mnist/test.csv'

if len(sys.argv) > 1:
    train_file = sys.argv[1]
    test_file = sys.argv[2]

def load_data(file):
    data = pd.read_csv(file, header=None)
    y = np.array(data[data.shape[1] - 1])
    y = y.reshape(-1, 1)
    x = np.array(data.drop(data.shape[1] - 1, axis=1))
    x = x / 255.0
    return x, y

C = [1e-5, 1e-3, 1, 5, 10]
X, Y = load_data(train_file)
testX, testY = load_data(test_file)
y1 = []
y2 = []
for c in C:
    print("Processing for C = " + str(c))
    m = len(X)
    best_acc = -1
    best_model = None
    avg_acc = 0
    for k in range(5):
        st = k*m//5
        en = min((k+1)*m//5, m)
        X_train = np.vstack((X[:st, :], X[en:, :]))
        Y_train = np.vstack((Y[:st, :], Y[en:, :]))
        X_valid = X[st:en, :]
        Y_valid = Y[st:en, :]
        prob = svm_train(Y_train.reshape(-1), X_train, "-c " + str(c) + " -s 0 -t 2 -g 0.05 -q")
        acc = svm_predict(Y_valid.reshape(-1), X_valid, prob)[1][0]
        avg_acc += acc
        if acc > best_acc:
            best_acc = acc
            best_model = prob
    TestAcc = svm_predict(testY.reshape(-1), testX, prob)[1][0]
    avg_acc /= 5
    print("C =", c, "Test Accuracy =", TestAcc, "Best Validation Accuracy =", best_acc, "Average Validation Accuracy =", avg_acc)
    y1.append(TestAcc)
    y2.append(avg_acc)

c = [0.00001, 0.001, 1, 5, 10]
logc = [-5, -3, 0, 0.6989, 1]
plt.scatter(logc, y1, s=30, label="Test Accuracy")
plt.scatter(logc, y2, s=8, label="5 Fold Validation Accuracy")
plt.xlabel("LogC")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("Accuracy_vs_logc_plot.png")
