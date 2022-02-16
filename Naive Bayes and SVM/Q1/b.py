import random, sys
import numpy as np
import pandas as pd

train_file = "/Users/aparahuja/Desktop/IITD/ML/Assignment 2/Q1/reviews_Digital_Music_5.json/Music_Review_train.json"
test_file = "/Users/aparahuja/Desktop/IITD/ML/Assignment 2/Q1/reviews_Digital_Music_5.json/Music_Review_test.json"


if len(sys.argv) > 1:
    train_file = sys.argv[1]
    test_file = sys.argv[2]

train_data = pd.read_json(train_file, lines=True)
test_data = pd.read_json(test_file, lines=True)

phi = {}
m = len(train_data)

def F1_Confusion(y_true, y_predict):
    confusion = np.zeros((5, 5))
    n = len(y_true)
    for i in range(n):
        confusion[int(y_true[i]) - 1][int(y_predict[i]) - 1] += 1
    # print("Confusion Matrix: ")
    # print(confusion)
    print("F1 scores:")
    f1_avg = 0
    for i in range(5):
        tp = confusion[i][i]
        fn = sum([confusion[j][i] if i != j else 0 for j in range(5)])
        fp = sum([confusion[i][j] if i != j else 0 for j in range(5)])
        f1_score = tp/(tp+(fp+fn)/2)
        f1_avg += f1_score/5
        print("\tClass " + str(i+1) + " = " + "{:.5f}".format(f1_score))
    print("Macro F1 score = " + "{:.5f}".format(f1_avg))

def initializeModel():
    for index, data in train_data.iterrows():
        label = data['overall']
        phi[label] = phi.get(label, 0) + 1


def findRandomModelAccuracy(input_data, datatype):
    print("Running randomized model on " + datatype + " data.")
    correct = 0
    total = len(input_data)
    labels = list(phi.keys())
    y_true = []
    y_predict = []
    for index, data in input_data.iterrows():
        ans_label = data['overall']
        ans = random.choice(labels)
        if ans_label == ans:
            correct += 1
        y_true.append(ans_label)
        y_predict.append(ans)
    F1_Confusion(y_true, y_predict)
    print("Random Model " + datatype + " Accuracy:",
          "{:.2f}".format(correct/total*100) + "%")


def findMaxModelAccuracy(input_data, datatype):
    print("Running max freq model on " + datatype + " data.")
    correct = 0
    total = len(input_data)
    y_true = []
    y_predict = []
    maxFreqLabel = max(phi.keys(), key=lambda x: phi[x])
    for index, data in input_data.iterrows():
        ans_label = data['overall']
        ans = maxFreqLabel
        if ans_label == ans:
            correct += 1
        y_true.append(ans_label)
        y_predict.append(ans)
    F1_Confusion(y_true, y_predict)
    print("Max Freq Model " + datatype + " Accuracy:", "{:.2f}".format(correct/total*100) + "%")

initializeModel()

# findRandomModelAccuracy(train_data, "Training")
findRandomModelAccuracy(test_data, "Testing")

# findMaxModelAccuracy(train_data, "Training")
findMaxModelAccuracy(test_data, "Testing")
