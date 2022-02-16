import sys
from sklearn.neural_network import MLPClassifier
from Utility import *
import pandas as pd

HIDDEN_LAYER_ARCH = [100, 100]
TARGET_CLASSES = 10
FEATURES = 85
ACTIVATION_FUCTION = 1  # 1 for sigmoid 2 for ReLU
LR = 0.1
REGULARIZATION = 0
ITERATIONS = 3000
BATCH_SIZE = 100

training_data_file = "poker-hand-training-true.csv"
testing_data_file = "poker-hand-testing.csv"

if len(sys.argv) > 1:
    training_data_file = sys.argv[1]
    testing_data_file = sys.argv[2]
training_data = pd.read_csv(training_data_file,header=None)
testing_data = pd.read_csv(testing_data_file,header=None)

training_data = training_data.to_numpy()
testing_data = testing_data.to_numpy()

X_train, Y_train = one_hot_encoder(training_data)
X_test, Y_test = one_hot_encoder(testing_data)

hidden_layer_arch = tuple(HIDDEN_LAYER_ARCH)
activation = 'logistic' if ACTIVATION_FUCTION == 1 else 'relu'

classifier = MLPClassifier(hidden_layer_sizes=hidden_layer_arch, activation=activation, solver='sgd',
                           learning_rate='adaptive', learning_rate_init=LR, batch_size=BATCH_SIZE, max_iter=ITERATIONS)
classifier.fit(X_train,Y_train)

y = classifier.predict(X_train)
y_pred_test = classifier.predict(X_test)

y_dash = convert_output_to_oht(y)

accuracy = calc_accuracy(y, Y_train)
test_accuracy = calc_accuracy(y_pred_test, Y_test)
print("Activation is :", activation)
print(f'The Training Accuracy is : {round(accuracy,2)}%')
print("Training Confusion Matrix")
print(create_confusion_matrix(y,Y_train))
print(f'The Testing Accuracy is : {round(test_accuracy,2)}%')
print("Testing Confusion Matrix")
print(create_confusion_matrix(y_pred_test,Y_test))

print("___________________________________________________________________________")

ACTIVATION_FUCTION = 2
activation = 'logistic' if ACTIVATION_FUCTION == 1 else 'relu'

classifier = MLPClassifier(hidden_layer_sizes=hidden_layer_arch, activation=activation, solver='sgd',
                           learning_rate='adaptive', learning_rate_init=LR, batch_size=BATCH_SIZE, max_iter=ITERATIONS)
classifier.fit(X_train, Y_train)

y = classifier.predict(X_train)
y_pred_test = classifier.predict(X_test)

y_dash = convert_output_to_oht(y)

accuracy = calc_accuracy(y, Y_train)
test_accuracy = calc_accuracy(y_pred_test, Y_test)
print("Activation is :", activation)
print(f'The Training Accuracy is : {round(accuracy,2)}%')
print("Training Confusion Matrix")
print(create_confusion_matrix(y, Y_train))
print(f'The Testing Accuracy is : {round(test_accuracy,2)}%')
print("Testing Confusion Matrix")
print(create_confusion_matrix(y_pred_test, Y_test))
