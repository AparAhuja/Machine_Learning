import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

train_file = "bank_dataset/bank_train.csv"
test_file = "bank_dataset/bank_test.csv"
val_file = "bank_dataset/bank_val.csv"

if len(sys.argv) > 1:
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    val_file = sys.argv[3]

train_data = pd.read_csv(train_file, delimiter=';')
val_data = pd.read_csv(val_file, delimiter=';')
test_data = pd.read_csv(test_file, delimiter=';')

attributes = list(train_data.columns)[:-1]
categorical = ["job", "marital", "education", "default",
               "housing", "loan", "contact", "month", "poutcome"]
categorical_cols = [attributes.index(x) for x in categorical]
uniqueVal = {}
X_train = train_data.loc[:, train_data.columns != 'y']
X_test = test_data.loc[:, test_data.columns != 'y']
X_val = val_data.loc[:, val_data.columns != 'y']

for category in categorical:
    uniqueVal[attributes.index(category)] = list(X_train[category].unique())

X_train = X_train.to_numpy()
Y_train = np.array([1 if y == 'yes' else 0 for y in train_data['y']])

X_test = X_test.to_numpy()
Y_test = np.array([1 if y == 'yes' else 0 for y in test_data['y']])

X_val = X_val.to_numpy()
Y_val = np.array([1 if y == 'yes' else 0 for y in val_data['y']])


class node:
    def __init__(self):
        self.children = {}
        self.attr = None
        self.threshold = None
        self.isNum = False
        self.isleaf = False
        self.pred = None
        self.depth = -1
        self.isPruned = False

def countNodes(root, ans = 0):
    if root == None:
        return 0
    if root.isPruned:
        return 1
    for key in root.children:
        ans += countNodes(root.children[key])
    return ans + 1

def H(y):
    pos = y[np.where(y == 1)[0]].shape[0]
    neg = y[np.where(y == 0)[0]].shape[0]
    if pos == 0 or neg == 0:
        return 0
    p0 = pos/(pos+neg)
    p1 = neg/(pos+neg)
    return - p0*np.log(p0) - p1*np.log(p1)


def H_j(x, y, j):
    ans = 0
    if j in categorical_cols:
        for value in uniqueVal[j]:
            px = np.count_nonzero(x[:, j] == value) / len(x)
            if px == 0:
                continue
            y_x = y[np.where(x[:, j] == value)]
            py1_x = np.sum(y_x) / len(y_x)
            py0_x = 1 - py1_x
            if py1_x == 0 or py0_x == 0:
                continue
            ans -= px*(py0_x*np.log(py0_x) + py1_x*np.log(py1_x))
    else:
        mid = np.median(x[:, j], axis=0)
        lessMid = y[np.where(x[:, j] <= mid)]
        moreMid = y[np.where(x[:, j] > mid)]
        py1_xless = sum(lessMid)/len(lessMid)
        py0_xless = 1 - py1_xless
        if py1_xless != 0 and py1_xless != 1:
            ans -= len(lessMid)/len(x)*(py1_xless *
                                        np.log(py1_xless) + py0_xless * np.log(py0_xless))
        if len(moreMid) > 0:
            py1_xmore = sum(moreMid)/len(moreMid)
            py0_xmore = 1 - py1_xmore
            if py1_xmore != 0 and py1_xmore != 1:
                ans -= len(moreMid)/len(x)*(py1_xmore *
                                            np.log(py1_xmore) + py0_xmore * np.log(py0_xmore))
    return ans


def best_attribute(x, y):
    max_MI = - sys.maxsize
    bAtr = - 1
    for i in range(len(attributes)):
        MI = H(y) - H_j(x, y, i)
        if MI > max_MI:
            bAtr = i
            max_MI = MI
    threshold = None
    if bAtr not in categorical_cols:
        threshold = np.median(x[:, bAtr], axis=0)
    return bAtr, threshold


def predictionHelper(x, root):
    if root.isleaf or root.isPruned:
        return root.pred
    if root.isNum:
        if x[root.attr] <= root.threshold:
            if root.children[0] is not None:
                return predictionHelper(x, root.children[0])
            else:
                return root.pred
        else:
            if root.children[1] is not None:
                return predictionHelper(x, root.children[1])
            else:
                return root.pred
    else:
        if x[root.attr] in root.children and root.children[x[root.attr]] is not None:
            return predictionHelper(x, root.children[x[root.attr]])
        else:
            return root.pred


def predict(X, root):
    return [predictionHelper(x, root) for x in X]


def find_accuracy(y_pred, y_true):
    correct = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            correct += 1
    return correct/len(y_true)*100


node_cnt = 0

nodes = []

def makeTree(x, y, depth):
    global node_cnt, nodes
    node_cnt += 1
    positive = sum(y)
    negative = len(y) - sum(y)
    if positive == 0 and negative == 0:
        # print("None")
        return None
    if positive == 0 or negative == 0:
        leaf = node()
        leaf.isleaf = True
        leaf.pred = 1 if positive > negative else 0
        # print("leaf", x, y)
        leaf.depth = depth
        return leaf
    bAtr, threshold = best_attribute(x, y)
    curr_node = node()
    curr_node.attr = bAtr
    curr_node.threshold = threshold
    curr_node.pred = 1 if positive > negative else 0
    if bAtr in categorical_cols:
        for value in uniqueVal[bAtr]:
            x_split = x[np.where(x[:, bAtr] == value)]
            y_split = y[np.where(x[:, bAtr] == value)]
            curr_node.children[value] = makeTree(x_split, y_split, depth + 1)
    else:
        curr_node.isNum = True
        x_split_0 = x[np.where(x[:, bAtr] <= threshold)]
        y_split_0 = y[np.where(x[:, bAtr] <= threshold)]
        curr_node.children[0] = makeTree(x_split_0, y_split_0, depth + 1)
        x_split_1 = x[np.where(x[:, bAtr] > threshold)]
        y_split_1 = y[np.where(x[:, bAtr] > threshold)]
        curr_node.children[1] = makeTree(x_split_1, y_split_1, depth + 1)
    # print(attributes[curr_node.attr], x, y)
    curr_node.depth = depth
    nodes.append(curr_node)
    return curr_node


numberOfNodesList = []
testAccuracyList = []
trainAccuracyList = []
valAccuracyList = []
root = makeTree(X_train, Y_train, 0)
nodes.sort(key=lambda x: x.depth)
nodes.pop(0)

prediction = predict(X_train, root)
accuracy = find_accuracy(prediction, Y_train)
trainAccuracyList.append(accuracy)

prediction = predict(X_val, root)
accuracy = find_accuracy(prediction, Y_val)
valAccuracyList.append(accuracy)

prediction = predict(X_test, root)
accuracy = find_accuracy(prediction, Y_test)
testAccuracyList.append(accuracy)

numberOfNodesList.append(node_cnt)

best_acc = valAccuracyList[-1]

for i in range(len(nodes) - 1, -1, -1):
    nodes[i].isPruned = True
    prediction = predict(X_val, root)
    curr_acc = find_accuracy(prediction, Y_val)
    nodes[i].isPruned = False
    if curr_acc > best_acc:
        best_acc = curr_acc
        nodes[i].isPruned = True

        numberOfNodesList.append(countNodes(root))

        prediction = predict(X_train, root)
        accuracy = find_accuracy(prediction, Y_train)
        trainAccuracyList.append(accuracy)

        prediction = predict(X_val, root)
        accuracy = find_accuracy(prediction, Y_val)
        valAccuracyList.append(accuracy)

        prediction = predict(X_test, root)
        accuracy = find_accuracy(prediction, Y_test)
        testAccuracyList.append(accuracy)

        print("Number of nodes =", numberOfNodesList[-1])
        print("Validation Accuracy =", valAccuracyList[-1])

# Removing best node pruned the root itself so I replaced the pruning method with a bottom up way in the code above
# while True:
#     print("Saving Accuracies ... ")
#     prediction = predict(X_train, root)
#     accuracy = find_accuracy(prediction, Y_train)
#     trainAccuracyList.append(accuracy)

#     prediction = predict(X_val, root)
#     accuracy = find_accuracy(prediction, Y_val)
#     valAccuracyList.append(accuracy)
#     print("Number of node =", node_cnt)
#     print("Testing accuracy =", "{:.2f}".format(accuracy), "%")

#     prediction = predict(X_test, root)
#     accuracy = find_accuracy(prediction, Y_test)
#     testAccuracyList.append(accuracy)

#     numberOfNodesList.append(node_cnt)

#     best_acc = -1
#     best_node = -1

#     print("Starting pruning ... ")
#     for i in range(len(nodes) - 1, -1, -1):
#         nodes[i].isPruned = True
#         prediction = predict(X_val, root)
#         curr_acc = find_accuracy(prediction, Y_val)
#         nodes[i].isPruned = False
#         if curr_acc > best_acc:
#             best_acc = curr_acc
#             best_node = i
#     print("Pruning complete ... ")
#     print(root.isPruned)
#     if best_acc < valAccuracyList[-1]:
#         break
#     nodes[best_node].isPruned = True
#     nodes.pop(best_node)
#     node_cnt = countNodes(root)


plt.plot(numberOfNodesList, testAccuracyList, label="Testing Accuracy")
plt.plot(numberOfNodesList, trainAccuracyList, label="Training Accuracy")
plt.plot(numberOfNodesList, valAccuracyList, label="Validation Accuracy")
plt.title("Multi Way Split Pruning")
plt.xlabel("Number of Nodes")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.5)
plt.show()

plt.close()

print("___________________________________________")

attributes = list(train_data.columns)[:-1]
categorical = ["job", "marital", "education", "default",
               "housing", "loan", "contact", "month", "poutcome"]
categorical_cols = [attributes.index(x) for x in categorical]
uniqueVal = {}
X_train = train_data.loc[:, train_data.columns != 'y']
X_test = test_data.loc[:, test_data.columns != 'y']
X_val = val_data.loc[:, val_data.columns != 'y']

new_categorical = []
cnt = X_train.shape[1]
for category in categorical:
    cnt -= 1
    one_hot_df = pd.get_dummies(X_train[category])
    X_train = X_train.drop(category, axis=1)
    X_train = X_train.join(one_hot_df, rsuffix="_" + category)

    one_hot_df = pd.get_dummies(X_test[category])
    X_test = X_test.drop(category, axis=1)
    X_test = X_test.join(one_hot_df, rsuffix="_" + category)

    one_hot_df = pd.get_dummies(X_val[category])
    X_val = X_val.drop(category, axis=1)
    X_val = X_val.join(one_hot_df, rsuffix="_" + category)

categorical_cols = [i for i in range(cnt, X_train.shape[1])]
categorical = list(X_train.columns)[cnt:]
attributes = list(X_train.columns)

for category in categorical:
    uniqueVal[attributes.index(category)] = list(X_train[category].unique())

X_train = X_train.to_numpy()
Y_train = np.array([1 if y == 'yes' else 0 for y in train_data['y']])

X_test = X_test.to_numpy()
Y_test = np.array([1 if y == 'yes' else 0 for y in test_data['y']])

X_val = X_val.to_numpy()
Y_val = np.array([1 if y == 'yes' else 0 for y in val_data['y']])

node_cnt = 0
nodes = []
numberOfNodesList = []
testAccuracyList = []
trainAccuracyList = []
valAccuracyList = []
root = makeTree(X_train, Y_train, 0)
nodes.sort(key=lambda x: x.depth)
nodes.pop(0)

prediction = predict(X_train, root)
accuracy = find_accuracy(prediction, Y_train)
trainAccuracyList.append(accuracy)

prediction = predict(X_val, root)
accuracy = find_accuracy(prediction, Y_val)
valAccuracyList.append(accuracy)

prediction = predict(X_test, root)
accuracy = find_accuracy(prediction, Y_test)
testAccuracyList.append(accuracy)

numberOfNodesList.append(node_cnt)

best_acc = valAccuracyList[-1]

for i in range(len(nodes) - 1, -1, -1):
    nodes[i].isPruned = True
    prediction = predict(X_val, root)
    curr_acc = find_accuracy(prediction, Y_val)
    nodes[i].isPruned = False
    if curr_acc > best_acc:
        best_acc = curr_acc
        nodes[i].isPruned = True

        numberOfNodesList.append(countNodes(root))

        prediction = predict(X_train, root)
        accuracy = find_accuracy(prediction, Y_train)
        trainAccuracyList.append(accuracy)

        prediction = predict(X_val, root)
        accuracy = find_accuracy(prediction, Y_val)
        valAccuracyList.append(accuracy)

        prediction = predict(X_test, root)
        accuracy = find_accuracy(prediction, Y_test)
        testAccuracyList.append(accuracy)

        print("Number of nodes =", numberOfNodesList[-1])
        print("Validation Accuracy =", valAccuracyList[-1])


plt.plot(numberOfNodesList, testAccuracyList, label="Testing Accuracy")
plt.plot(numberOfNodesList, trainAccuracyList, label="Training Accuracy")
plt.plot(numberOfNodesList, valAccuracyList, label="Validation Accuracy")
plt.title("One Hot Encoding Pruning")
plt.xlabel("Number of Nodes")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.5)
plt.show()
