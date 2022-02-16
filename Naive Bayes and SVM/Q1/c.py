import sys, time
import json
import string
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd

train_file = "/Users/aparahuja/Desktop/IITD/ML/Assignment 2/Q1/reviews_Digital_Music_5.json/Music_Review_train.json"
test_file = "/Users/aparahuja/Desktop/IITD/ML/Assignment 2/Q1/reviews_Digital_Music_5.json/Music_Review_test.json"


if len(sys.argv) > 1:
    train_file = sys.argv[1]
    test_file = sys.argv[2]

train_data = pd.read_json(train_file, lines=True)
test_data = pd.read_json(test_file, lines=True)

vocabulary = set()
theta = {}
phi = {}
cnt = {}
m = len(train_data)


def Confusion(y_true, y_predict):
    confusion = np.zeros((5, 5))
    n = len(y_true)
    for i in range(n):
        confusion[int(y_true[i]) - 1][int(y_predict[i]) - 1] += 1
    print("Confusion Matrix: ")
    print(confusion)
    maxDiagonal = max([i for i in range(5)], key = lambda x : confusion[x][x])
    print("Max Diagonal Entry in confusion matrix occurs for the class - " + str(maxDiagonal + 1))


def tokenize(review):
    tokens = [w.lower() for w in word_tokenize(review)]
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    return [word for word in stripped if word.isalpha()]

TokenizedReview = []
def initializeModel():
    global vocabulary
    for index, data in train_data.iterrows():
        review = data['reviewText']
        label = data['overall']
        theta[label] = {}
        cnt[label] = 0
        phi[label] = 0
        words = tokenize(review)
        TokenizedReview.append(words)
        for word in words:
            vocabulary.add(word)

    vocabulary.add("UNK")

    file = open("vocabulary_a.txt", "w")
    file.write(json.dumps(list(vocabulary)))
    file.close()

    for label in theta:
        for word in vocabulary:
            theta[label][word] = 0


def learnParameters():
    for index, data in train_data.iterrows():
        review = data['reviewText']
        label = data['overall']
        # words = tokenize(review)
        words = TokenizedReview[index]

        phi[label] += 1
        for word in words:
            theta[label][word] += 1
            cnt[label] += 1

    for label in theta:
        for word in vocabulary:
            theta[label][word] = (theta[label][word] + 1) / \
                (cnt[label] + len(vocabulary) + 1)
        phi[label] /= m


def predict(words, label):
    ans = np.log(phi[label])
    for word in words:
        if word in vocabulary:
            ans += np.log(theta[label][word])
        else:
            ans += np.log(theta[label]["UNK"])
    return ans


def findModelAccuracy(input_data, datatype):
    print("Running model on " + datatype + " data.")
    correct = 0
    total = len(input_data)
    y_true = []
    y_predict = []
    for index, data in input_data.iterrows():
        review = data['reviewText']
        words = tokenize(review)
        ans_label = data['overall']
        ans, logProbab = "", -sys.maxsize
        for label in phi:
            prediction = predict(words, label)
            if logProbab <= prediction:
                ans = label
                logProbab = prediction
        if ans_label == ans:
            correct += 1
        y_true.append(ans_label)
        y_predict.append(ans)
    Confusion(y_true, y_predict)
    print("Model " + datatype + " accuracy:",
          "{:.2f}".format(correct/total*100) + "%")


st = time.time()
initializeModel()
learnParameters()
en = time.time()
print("Training Time = " + "{:.2f}".format(en - st) + " sec")

# findModelAccuracy(train_data, "training")
findModelAccuracy(test_data, "testing")
