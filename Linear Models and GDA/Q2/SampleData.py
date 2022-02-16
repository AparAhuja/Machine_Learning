import numpy as np

def sampleData():
    data_size = 1000000
    f = open("data/q2/q2train.csv", 'w')
    f.write("X_1,X_2,Y\n")
    X1 = np.random.normal(3, 2, (data_size, 1))
    X2 = np.random.normal(-1, 2, (data_size, 1))
    ones = np.ones((data_size, 1))
    X = np.append(np.append(X2, X1, axis=1), ones, axis=1)
    theta = np.array([[2], [1], [3]])
    noise = np.random.normal(0, np.sqrt(2), (data_size, 1))
    Y = np.dot(X, theta) + noise
    for i in range(data_size):
        f.write(str(X1[i][0])+","+str(X2[i][0])+","+str(Y[i][0])+"\n")
    return X, Y

sampleData()