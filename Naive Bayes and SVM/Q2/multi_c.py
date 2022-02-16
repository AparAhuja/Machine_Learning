from PIL import Image
import time
from libsvm.svmutil import svm_predict, svm_train
import numpy as np
import sys
import pandas as pd
from libsvm.svm import *
from libsvm.svmutil import *

train_file = '/Users/aparahuja/Desktop/IITD/ML/Assignment 2/Q2/mnist/train.csv'
test_file = '/Users/aparahuja/Desktop/IITD/ML/Assignment 2/Q2/mnist/test.csv'

if len(sys.argv) > 1:
    train_file = sys.argv[1]
    test_file = sys.argv[2]


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


def Binary_F1_Confusion(y_true, y_predict):
    confusion = np.zeros((2, 2))
    n = len(y_true)
    for i in range(n):
        confusion[int(y_true[i])][int(y_predict[i])] += 1
    print("Confusion Matrix: ")
    print(confusion)
    maxDiagonal = max([i for i in range(2)], key=lambda x: confusion[x][x])
    print("Max Diagonal Entry in confusion matrix occurs for the class - " + str(maxDiagonal))
    print("F1 scores:")
    f1_avg = 0
    for i in range(2):
        tp = confusion[i][i]
        fn = sum([confusion[j][i] if i != j else 0 for j in range(2)])
        fp = sum([confusion[i][j] if i != j else 0 for j in range(2)])
        f1_score = tp/(tp+(fp+fn)/2)
        f1_avg += f1_score/2
        print("\tClass " + str(i) + " = " + "{:.5f}".format(f1_score))
    print("Macro F1 score = " + "{:.5f}".format(f1_avg))


def load_data(file):
    data = pd.read_csv(file, header=None)
    y = np.array(data[data.shape[1] - 1])
    y = y.reshape(-1, 1)
    x = np.array(data.drop(data.shape[1] - 1, axis=1))
    x = x / 255.0
    return x, y


def load_data_binary(file, a, b):
    a, b = a % 10, b % 10
    data_all = pd.read_csv(file, header=None)
    data = data_all[(data_all[data_all.shape[1]-1] == a) |
                    (data_all[data_all.shape[1] - 1] == b)]
    y = np.array(data[data_all.shape[1] - 1])
    y = y.reshape(-1, 1)
    y[y == a] = 1
    y[y == b] = 0
    x = np.array(data.drop(data_all.shape[1] - 1, axis=1))
    x = x / 255.0
    return x, y

print("MULTI-CLASS CLASSIFICATION MODEL")
X, Y = load_data(train_file)
testX, testY = load_data(test_file)

st = time.time()
prob = svm_train(Y.reshape(-1), X, "-s 0 -t 2 -g 0.05 -q")
en = time.time()
print("Training Time = " + "{:.2f}".format(en - st) + " sec")

print("Testing -")
predlabel = svm_predict(testY.reshape(-1), testX, prob)[0]
F1_Confusion(testY.reshape(-1), predlabel)

print("BINARY CLASSIFICATION MODEL")
X, Y = load_data_binary(train_file, 5, 6)
testX, testY = load_data_binary(test_file, 5, 6)

st = time.time()
prob = svm_train(Y.reshape(-1), X, "-s 0 -t 2 -g 0.05 -q")
en = time.time()
print("Training Time = " + "{:.2f}".format(en - st) + " sec")

print("Testing -")
predlabel = svm_predict(testY.reshape(-1), testX, prob)[0]
Binary_F1_Confusion(testY.reshape(-1), predlabel)

g = open("misclassified.txt", 'w')
g.close()
f = open("misclassified.txt", 'a')
for i in range(len(predlabel)):
    if predlabel[i] != testY[i][0]:
        for x in X[i][:-1]:
            f.write(str(int(255*x)) + ",")
        f.write(str(int(255*X[i][-1]))+"\n")
f.close()

f = open("misclassified.txt", 'r')
lines = [(l.strip('\n')).split(",") for l in f.readlines()]
lines = [[int(x) for x in line] for line in lines]

cnt = 0

for line in lines:
    cnt += 1
    data = np.zeros((28, 28), dtype=np.uint8)
    for i in range(28):
        for j in range(28):
            data[i][j] = line[28*i + j]
    img = Image.fromarray(data, 'L')
    img.save(str(cnt) + '.png')
    # comment the lines below to store all misclassified digits as images
    if cnt > 10:
        break
