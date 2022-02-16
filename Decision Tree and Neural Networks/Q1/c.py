import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
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

grid = {'n_estimators': [i for i in range(50, 451, 100)],
        'max_features': [0.1, 0.3, 0.5, 0.7, 0.9],
        'min_samples_split': [i for i in range(2, 11, 2)]}

def score(est, x, y):
    est.fit(x, y)
    return est.oob_score_

forest = RandomForestClassifier(criterion="entropy", oob_score=True)
search = GridSearchCV(forest, grid, scoring=score, n_jobs=-1)
result = search.fit(X_train, Y_train.to_numpy().reshape((-1,)))
print("Best Out-of-bag Accuracy =", 100*result.best_score_)
print("Best Parameters =", result.best_params_)

print("______________________________________\n")

best_model = result.best_estimator_

testAccuracy = best_model.score(X_test, Y_test.to_numpy().reshape((-1,)))
print("Test Accuracy =", "{:.2f}".format(100*testAccuracy), "%")
trainAccuracy = best_model.score(X_train, Y_train.to_numpy().reshape((-1,)))
print("Train Accuracy =", "{:.2f}".format(100*trainAccuracy), "%")
valAccuracy = best_model.score(X_val, Y_val.to_numpy().reshape((-1,)))
print("Validation Accuracy =", "{:.2f}".format(100*valAccuracy), "%")

# Best Out-of-bag Accuracy = 90.38374901444899
# Best Parameters = {'max_features': 0.3, 'min_samples_split': 10, 'n_estimators': 350}
