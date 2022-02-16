import sys
import numpy as np
import pandas as pd
import math
from Utility import *
import time

HIDDEN_LAYER_ARCH = [100, 100]
TARGET_CLASSES = 10
FEATURES = 85
ACTIVATION_FUCTION = 1 # 1 for sigmoid 2 for ReLU
LR = 3
REGULARIZATION = 0
ITERATIONS = 3000
BATCH_SIZE = 100
LR_DECAY = True

training_data_file = "poker-hand-training-true.csv"
testing_data_file = "poker-hand-testing.csv"


if len(sys.argv) > 1:
    training_data_file = sys.argv[1]
    testing_data_file = sys.argv[2]
training_data = pd.read_csv(training_data_file, header=None)
testing_data = pd.read_csv(testing_data_file, header=None)


def find_activation_derivative(activations):
    if ACTIVATION_FUCTION == 1:
        return np.multiply(activations, 1-activations)
    elif ACTIVATION_FUCTION == 2:
        return ReLU_der(activations)
    else:
        raise Exception(
            "Activation Not Defined! Use 1 for sigmoid and 2 for ReLU")


def apply_activation(x):
    if ACTIVATION_FUCTION == 1:
        return sigmoid(x)
    elif ACTIVATION_FUCTION == 2:
        return ReLU(x)
    else:
        raise Exception(
            "Activation Not Defined! Use 1 for sigmoid and 2 for ReLU")


def initialize_weights():
    LAYERS = [FEATURES] + HIDDEN_LAYER_ARCH + [TARGET_CLASSES]
    W = []
    for i in range(len(LAYERS)-1):
        epsilon = math.sqrt((LAYERS[i]+LAYERS[i+1])/1000)
        if ACTIVATION_FUCTION == 2:
            W_temp = np.random.normal(size = (LAYERS[i]+1, LAYERS[i+1]))*np.sqrt(2 / LAYERS[i+1])
        else:
            W_temp = np.random.normal(size=(LAYERS[i]+1, LAYERS[i+1]))
        W_temp = np.random.normal(size=(LAYERS[i]+1, LAYERS[i+1])) * np.sqrt(2 / LAYERS[i+1])
        if ACTIVATION_FUCTION == 1:
            W_temp = np.random.normal(size=(LAYERS[i]+1, LAYERS[i+1]))
        W.append(W_temp)
        print(f'shape of weight {i + 1} : ({len(W_temp)},{len(W_temp[0])})')
    return W


def forward_propogation(input, weights):
    activations = []
    batch_size = len(input)
    e = np.ones((batch_size, 1))
    activations.append(np.hstack([e, input]))
    for i in range(len(HIDDEN_LAYER_ARCH) + 1):
        if i != len(HIDDEN_LAYER_ARCH):
            temp_act = apply_activation(np.matmul(activations[i], weights[i]))
            # adding bias node to all activations except the last
            temp_act = np.hstack([e, temp_act])
        else:
            # Apply sigmoid for the last layer
            temp_act = sigmoid(np.matmul(activations[i], weights[i]))
        activations.append(temp_act)
    return activations


def back_propogation(activations, weights, expected_output, lr):
    delta = []
    batch_size = len(expected_output)
    delta.append(
        2*np.subtract(activations[len(activations)-1], expected_output))
    j = 0
    # The following code calculates the delta for all the layers (delta = dE/da)
    for i in range(len(weights)-1, -1, -1):
        if i == len(weights) - 1:  # Last layer
            derivative_sigmoid = np.multiply(
                activations[i+1], 1-activations[i+1])
            temp = np.multiply(delta[j], derivative_sigmoid)
            temp = np.matmul(temp, np.transpose(weights[i]))
        else:  # other than last layers
            derivative_activation = find_activation_derivative(
                activations[i+1])
            temp = np.multiply(delta[j], derivative_activation)
            temp = np.matmul(temp[:, 1:], np.transpose(weights[i]))
        delta.append(temp)
        j = j + 1
    k = 0
    # Following code is to update the weights using the deltas we have calculated (delta is dE/da for all activations layers)
    for i in range(len(weights)-1, -1, -1):
        if i == len(weights) - 1:  # Last Layer
            derivative_sigmoid = np.multiply(
                activations[i+1], 1-activations[i+1])
            gradient = np.multiply(delta[k], derivative_sigmoid)
            gradient = np.matmul(np.transpose(activations[i]), gradient)
        else:  # Other than last layer
            derivative_activation = find_activation_derivative(
                activations[i+1])
            gradient = np.multiply(
                delta[k][:, 1:], derivative_activation[:, 1:])
            gradient = np.matmul(np.transpose(activations[i]), gradient)

        gradient = gradient + REGULARIZATION*weights[i]
        gradient = gradient/(2*batch_size)
        weights[i] = weights[i] - lr*gradient
        k = k + 1
    return weights


start = time.time()
training_data = training_data.to_numpy()
testing_data = testing_data.to_numpy()
np.random.shuffle(training_data)
encoded_x, encoded_y = one_hot_encoder(training_data)
batched_x, batched_y = batchify(encoded_x, encoded_y)
weights = initialize_weights()
activations = None

MSE_List = []
k = len(batched_x)
cnt = 0
for i in range(ITERATIONS + 1):
    effective_lr = LR / math.sqrt(i+1) if LR_DECAY else LR
    if i % 100 == 0:
        activations = forward_propogation(encoded_x, weights)
        loading_bar = '#'*(i//50) + '-'*((ITERATIONS-i)//50)
        #print(f'epoch {i}:{loading_bar} Training Accuracy is {round(calc_accuracy(activations[len(activations)-1],encoded_y),2)}%')
    for batch_number in range(len(batched_x)):
        batched_activations = forward_propogation(
            batched_x[batch_number], weights)
        weights = back_propogation(
            batched_activations, weights, batched_y[batch_number], effective_lr)
        MSE_List.append(calculate_error(
            batched_activations[-1], batched_y[batch_number]))
    # STOPPING CRITERIA
    # Note that we iterate through the whole dataset atleast once above.
    if len(MSE_List) > 2*k and abs(sum(MSE_List[-1:-k-1:-1])/k - sum(MSE_List[-k-1:-2*k-1:-1])/k) < 1e-6:
        cnt += 1
        if cnt == 5:
            print("Stopping criteria satisfied.")
            print("Number of epochs is", i)
            print("Ending training...")
            break

output = activations[len(activations)-1]
end = time.time()
if ACTIVATION_FUCTION == 1:
    print("\nActivation = Sigmoid\n")
else:
    print("\nActivation = ReLU\n")
print(f'Training Time : {round(end - start,2)}  secs')
print(f'Training MSE : {round(calculate_error(output,encoded_y),4)}')
print(f'Training Accuracy : {round(calc_accuracy(output,encoded_y),2)}%')
print("Training Confusion Matrix")
print(create_confusion_matrix(output, encoded_y))

print("-------------------------------------------------")
print("Testing...")

x_test, y_test = one_hot_encoder(testing_data)
test_activations = forward_propogation(x_test, weights)
test_output = test_activations[len(test_activations)-1]
print(f'Testing MSE : {round(calculate_error(test_output,y_test),4)}')
print(f'Testing Accuracy : {round(calc_accuracy(test_output,y_test),2)}%')
print("Testing Confusion Matrix")
print(create_confusion_matrix(test_output, y_test))


print("---------------------------------------------------------------------------------------------------------")
####################################################################################################################################################################################################################################


ACTIVATION_FUCTION = 2

start = time.time()
weights = initialize_weights()
activations = None

MSE_List = []
k = len(batched_x)
cnt = 0
for i in range(ITERATIONS + 1):
    effective_lr = LR / math.sqrt(i+1) if LR_DECAY else LR
    if i % 100 == 0:
        activations = forward_propogation(encoded_x, weights)
        loading_bar = '#'*(i//50) + '-'*((ITERATIONS-i)//50)
        #print(f'epoch {i}:{loading_bar} Training Accuracy is {round(calc_accuracy(activations[len(activations)-1],encoded_y),2)}%')
    for batch_number in range(len(batched_x)):
        batched_activations = forward_propogation(
            batched_x[batch_number], weights)
        weights = back_propogation(
            batched_activations, weights, batched_y[batch_number], effective_lr)
        MSE_List.append(calculate_error(
            batched_activations[-1], batched_y[batch_number]))
    # STOPPING CRITERIA
    # Note that we iterate through the whole dataset atleast once above.
    if len(MSE_List) > 2*k and abs(sum(MSE_List[-1:-k-1:-1])/k - sum(MSE_List[-k-1:-2*k-1:-1])/k) < 1e-6:
        cnt += 1
        if cnt == 5:
            print("Stopping criteria satisfied.")
            print("Number of epochs is", i)
            print("Ending training...")
            break

output = activations[len(activations)-1]
end = time.time()
if ACTIVATION_FUCTION == 1:
    print("\nActivation = Sigmoid\n")
else:
    print("\nActivation = ReLU\n")
print(f'Training Time : {round(end - start,2)}  secs')
print(f'Training MSE : {round(calculate_error(output,encoded_y),4)}')
print(f'Training Accuracy : {round(calc_accuracy(output,encoded_y),2)}%')
print("Training Confusion Matrix")
print(create_confusion_matrix(output, encoded_y))

print("-------------------------------------------------")
print("Testing...")

x_test, y_test = one_hot_encoder(testing_data)
test_activations = forward_propogation(x_test, weights)
test_output = test_activations[len(test_activations)-1]
print(f'Testing MSE : {round(calculate_error(test_output,y_test),4)}')
print(f'Testing Accuracy : {round(calc_accuracy(test_output,y_test),2)}%')
print("Testing Confusion Matrix")
print(create_confusion_matrix(test_output, y_test))
