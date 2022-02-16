import sys
import numpy as np
import pandas as pd

training_data_file = "poker-hand-training-true.csv"
testing_data_file = "poker-hand-testing.csv"

if len(sys.argv) > 1:
    training_data_file = sys.argv[1]
    testing_data_file = sys.argv[2]

training_data = pd.read_csv(training_data_file, header=None)
testing_data = pd.read_csv(testing_data_file, header=None)

def one_hot_encoder(data):
    encoded_x = np.zeros((len(data), 85))
    encoded_y = np.zeros((len(data), 10))
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

encoded_x, encoded_y = one_hot_encoder(training_data.to_numpy())
one_hot_data = np.hstack((encoded_x, encoded_y)).astype(np.int64)
one_hot_data = pd.DataFrame(one_hot_data)
one_hot_data.to_csv("poker-hand-training-true-one-hot.csv", header=None, index=False)

encoded_x, encoded_y = one_hot_encoder(testing_data.to_numpy())
one_hot_data = np.hstack((encoded_x, encoded_y)).astype(np.int64)
one_hot_data = pd.DataFrame(one_hot_data)
one_hot_data.to_csv("poker-hand-testing-one-hot.csv")
