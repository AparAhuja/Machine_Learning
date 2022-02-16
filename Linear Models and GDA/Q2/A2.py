import time
import numpy as np
import matplotlib.pyplot as plt

batch_size = 10000

def load_data(fileX):
    data = np.genfromtxt(fileX, delimiter=',')
    x = data[1:, 1::-1]
    y = data[1:, data.shape[1]-1:]
    ones = np.ones((x.shape[0], 1))
    x = np.append(x, ones, axis=1)
    return x, y


def cost(theta, X, Y):
    error = Y - np.dot(X, theta)
    sq_error = np.dot(error.T, error)[0][0]
    avg_cost = sq_error/(2*X.shape[0])
    return avg_cost


def update(theta, eta, X, Y):
    error = Y - np.dot(X, theta)
    theta_new = theta + eta*np.dot(X.T, error)/X.shape[0]
    return theta_new


def print_history(l):
    for x in l:
        print(x)


def plot(theta_history):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    theta2 = np.array([x[0][0] for x in theta_history])
    theta1 = np.array([x[1][0] for x in theta_history])
    theta0 = np.array([x[2][0] for x in theta_history])
    ax.plot3D(theta2, theta1, theta0, 'red')
    plt.show()


def testData(theta):
    X, Y = load_data("data/q2/q2test.csv")
    print("Learnt Hypothesis Cost: ", cost(theta, X, Y))
    print("Original Hypothesis Cost: ", cost(np.array([[2], [1], [3]]), X, Y))


def converged(cost_history):
    k = 2000
    if len(cost_history) < 2*k + 1:
        return False
    if abs(sum(cost_history[-1:-k-1:-1]) - sum(cost_history[-k-1:-2*k-1:-1]))/k < 10e-7:
        return True
    return False


X, Y = load_data("data/q2/q2train.csv")

theta = np.zeros((X.shape[1], 1))
eta = 0.001
batch_number = 0

theta_history = [theta]
cost_history = [cost(theta, X, Y)]

t = 0
prev_cost = -1
curr_cost = cost_history[0]

start_time = time.time()
while(not converged(cost_history)):
    sampleX = X[batch_number *
                batch_size:min(X.shape[0], (1+batch_number)*batch_size), :]
    sampleY = Y[batch_number *
                batch_size:min(X.shape[0], (1+batch_number)*batch_size), :]
    batch_number = batch_number + \
        1 if (1+batch_number)*batch_size < X.shape[0] else 0

    theta = update(theta, eta, sampleX, sampleY)

    prev_cost = curr_cost
    curr_cost = cost(theta, sampleX, sampleY)

    theta_history.append(theta)
    cost_history.append(curr_cost)

    t = t + 1
end_time = time.time()

print("Batch size:", batch_size)
print("Time take:", end_time - start_time)
print("Number of iterations:", t)
print("Theta [theta2, theta1, theta0] -\n", theta)
print("Final Sample Cost:", curr_cost)
testData(theta)
# plot(theta_history)
# print_history(theta_history)
# print_history(cost_history)
