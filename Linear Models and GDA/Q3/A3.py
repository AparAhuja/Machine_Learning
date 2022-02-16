import numpy as np
import matplotlib.pyplot as plt

def load_data(fileX, fileY):
    x = np.genfromtxt(fileX, delimiter=',')
    y = np.genfromtxt(fileY, delimiter=',')
    y = y.reshape((y.shape[0], 1))
    ones = np.ones((x.shape[0], 1))
    x = np.append(x, ones, axis=1)
    return x, y


def cost(theta, X, Y):
    h_theta = 1/(1+np.exp(-np.dot(X, theta)))
    log_liklihood = Y*np.log(h_theta) + (1-Y)*np.log(1-h_theta)
    return np.sum(log_liklihood)/(X.shape[0])


def update(theta, X, Y):
    h_theta = 1/(1+np.exp(-np.dot(X, theta)))
    gradient = np.dot(X.T, Y - h_theta)/X.shape[0]
    hessian = - np.dot(X.T, X*(h_theta*(1-h_theta)))/X.shape[0]
    theta_new = theta - np.dot(np.linalg.inv(hessian), gradient)
    return theta_new


def normalized(X):
    mean = X.mean(axis=0)
    std_dev = X.std(axis=0)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if(std_dev[j] == 0):
                continue
            X[i][j] -= mean[j]
            X[i][j] /= std_dev[j]
    return X

def plotb(X, Y, theta):
    class0 = []
    class1 = []
    for i in range(X.shape[0]):
        if Y[i][0] == 1:
            class1.append(X[i])
        else:
            class0.append(X[i])
    class0 = np.array(class0)
    class1 = np.array(class1)
    plt.scatter(class0[:, :1], class0[:, 1:2], 4, 'g', label = '0')
    plt.scatter(class1[:, :1], class1[:, 1:2], 4, 'b', label = '1')
    left, right = plt.xlim()
    x = np.linspace(left, right)
    y = (- theta[0][0]*x - theta[2][0]) / theta[1][0]
    plt.plot(x, y, 'r', label = 'Decision Boundry')
    plt.legend()
    plt.savefig("3b.png")

def print_history(l):
    for x in l:
        print(x)

X, Y = load_data("data/q3/logisticX.csv", "data/q3/logisticY.csv")
X = normalized(X)

theta = np.zeros((X.shape[1], 1))
theta_history = [theta]
cost_history = [cost(theta, X, Y)]

t = 0
prev_cost = - np.Infinity
curr_cost = cost_history[0]

while abs(curr_cost - prev_cost) > 10e-15:
    theta = update(theta, X, Y)

    prev_cost = curr_cost
    curr_cost = cost(theta, X, Y)

    theta_history.append(theta)
    cost_history.append(curr_cost)

    t = t + 1

print(t)
print(theta)
print(curr_cost)
plotb(X, Y, theta)
# print_history(theta_history)
# print_history(cost_history)
