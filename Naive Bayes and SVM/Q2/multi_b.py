from libsvm.svmutil import svm_predict, svm_train
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


X, Y = load_data(train_file)
prob = svm_train(Y.reshape(-1), X, "-s 0 -t 2 -g 0.05")
print("Training -")
svm_predict(Y.reshape(-1), X, prob)
testX, testY = load_data(test_file)
print("Testing -")
svm_predict(testY.reshape(-1), testX, prob)
