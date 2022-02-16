import sys
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd

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

X_train = train_data.loc[:, train_data.columns != 'y']
X_test = test_data.loc[:, test_data.columns != 'y']
X_val = val_data.loc[:, val_data.columns != 'y']

Y_train = train_data.loc[:, train_data.columns == 'y']
Y_test = test_data.loc[:, test_data.columns == 'y']
Y_val = val_data.loc[:, val_data.columns == 'y']

new_categorical = []
cnt = X_train.shape[1]
categorical = ["job", "marital", "education", "default",
               "housing", "loan", "contact", "month", "poutcome"]
for category in categorical:
    one_hot_df = pd.get_dummies(X_train[category])
    X_train = X_train.drop(category, axis=1)
    X_train = X_train.join(one_hot_df, rsuffix="_" + category)

    one_hot_df = pd.get_dummies(X_test[category])
    X_test = X_test.drop(category, axis=1)
    X_test = X_test.join(one_hot_df, rsuffix="_" + category)

    one_hot_df = pd.get_dummies(X_val[category])
    X_val = X_val.drop(category, axis=1)
    X_val = X_val.join(one_hot_df, rsuffix="_" + category)

print("Varying n_estimators")

parameterList = []
testAccuracyList = []
trainAccuracyList = []
valAccuracyList = []

for i in range(50, 451, 100):
    forest = RandomForestClassifier(n_estimators=i, max_features=0.3, min_samples_split=10, criterion="entropy")
    forest.fit(X_train, Y_train.to_numpy().reshape((-1,)))
    parameterList.append(i)
    trainAccuracyList.append(100*forest.score(X_train, Y_train.to_numpy().reshape((-1,))))
    testAccuracyList.append(100*forest.score(X_test, Y_test.to_numpy().reshape((-1,))))
    valAccuracyList.append(100*forest.score(X_val, Y_val.to_numpy().reshape((-1,))))

plt.plot(parameterList, testAccuracyList, label="Testing Accuracy")
plt.plot(parameterList, trainAccuracyList, label="Training Accuracy")
plt.plot(parameterList, valAccuracyList, label="Validation Accuracy")
plt.title("Varying n_estimators")
plt.xlabel("n_estimator")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.5)
plt.show()
plt.close()
print("_________________________________\n")


print("Varying max_features")

parameterList = []
testAccuracyList = []
trainAccuracyList = []
valAccuracyList = []

for i in [0.1, 0.3, 0.5, 0.7, 0.9]:
    forest = RandomForestClassifier(n_estimators=350, max_features=i, min_samples_split=10, criterion="entropy")
    forest.fit(X_train, Y_train.to_numpy().reshape((-1,)))
    parameterList.append(i)
    trainAccuracyList.append(100*forest.score(X_train, Y_train.to_numpy().reshape((-1,))))
    testAccuracyList.append(100*forest.score(X_test, Y_test.to_numpy().reshape((-1,))))
    valAccuracyList.append(100*forest.score(X_val, Y_val.to_numpy().reshape((-1,))))

plt.plot(parameterList, testAccuracyList, label="Testing Accuracy")
plt.plot(parameterList, trainAccuracyList, label="Training Accuracy")
plt.plot(parameterList, valAccuracyList, label="Validation Accuracy")
plt.title("Varying max_features")
plt.xlabel("max_features")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.5)
plt.show()

plt.close()
print("_________________________________\n")

print("Varying min_samples_split")

parameterList = []
testAccuracyList = []
trainAccuracyList = []
valAccuracyList = []

for i in range(2, 11, 2):
    forest = RandomForestClassifier(n_estimators=350, max_features=0.3, min_samples_split=i, criterion="entropy")
    forest.fit(X_train, Y_train.to_numpy().reshape((-1,)))
    parameterList.append(i)
    trainAccuracyList.append(100*forest.score(X_train, Y_train.to_numpy().reshape((-1,))))
    testAccuracyList.append(100*forest.score(X_test, Y_test.to_numpy().reshape((-1,))))
    valAccuracyList.append(100*forest.score(X_val, Y_val.to_numpy().reshape((-1,))))

plt.plot(parameterList, testAccuracyList, label="Testing Accuracy")
plt.plot(parameterList, trainAccuracyList, label="Training Accuracy")
plt.plot(parameterList, valAccuracyList, label="Validation Accuracy")
plt.title("Varying min_samples_split")
plt.xlabel("min_samples_split")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.5)
plt.show()

print("_________________________________\n")
