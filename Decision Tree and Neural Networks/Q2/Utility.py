import numpy as np
import sys

BATCH_SIZE = 100
FEATURES = 85
TARGET_CLASSES = 10

def sigmoid(z):
    y = 1/(1+np.exp(-1*z))
    return y

def ReLU(z):
    y = (np.abs(z) + z)/2
    return y

def ReLU_der(x):
    return (x > 0).astype(int)

def one_hot_encoder(data):
    encoded_x = np.zeros((len(data), FEATURES))
    encoded_y = np.zeros((len(data), TARGET_CLASSES))
    for data_index, data_point in enumerate(data):
        i = 0
        j = 0
        while i < 10:
            encoded_x[data_index][17*j + data_point[i] - 1] = 1
            encoded_x[data_index][17*j + 4 + data_point[i+1] - 1] = 1
            i = i + 2
            j = j + 1
        encoded_y[data_index][data_point[10]] = 1
    return encoded_x, encoded_y

def convert_output_to_oht(output):
    encoded_output = np.zeros(output.shape)
    for i in range(len(output)):
        maximum = -1*sys.maxsize
        j_star = 0
        for j in range(len(output[i])):
            if maximum < output[i][j]:
                maximum = output[i][j]
                j_star = j
        encoded_output[i][j_star] = 1
    return encoded_output

def calc_accuracy(output, expected_output):
    encoded_output = convert_output_to_oht(output)
    correct_label = 0
    for i in range(len(expected_output)):
        if (expected_output[i] == encoded_output[i]).all():
            correct_label += 1
    total_label = len(output)
    return 100*correct_label/total_label

def calculate_error(output,expected_output):
    #This function is to calculate the MSE
    difference = np.subtract(output,expected_output)
    squared_difference = np.multiply(difference,difference)
    error = np.sum(squared_difference)
    batch_size = len(output)
    return error/(2*batch_size)
        
def batchify(x,y):
    batched_x = [x[i:i + BATCH_SIZE] for i in range(0, len(x), BATCH_SIZE)] 
    batched_y = [y[i:i + BATCH_SIZE] for i in range(0, len(y), BATCH_SIZE)] 
    return batched_x, batched_y

def create_confusion_matrix(output, expected_output):
    encoded_output = convert_output_to_oht(output)
    return np.array(np.matmul(np.transpose(encoded_output),expected_output),dtype=int)

