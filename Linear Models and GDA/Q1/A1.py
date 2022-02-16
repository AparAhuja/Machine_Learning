import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

eta = 1

def load_data(fileX, fileY):
    x = np.genfromtxt(fileX, delimiter=',')
    y = np.genfromtxt(fileY, delimiter=',')
    x = x.reshape((x.shape[0], 1))
    y = y.reshape((y.shape[0], 1))
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


def print_history(l):
    for x in l:
        print(x)


def plotb(X, Y, theta):
    plt.scatter(X[:, :1], Y, s=4)
    left, right = plt.xlim()
    x = np.linspace(left, right)
    y = theta[0][0]*x + theta[1][0]
    plt.plot(x, y, 'r', label="Hypothesis Function")
    plt.xlabel("Normalized X")
    plt.ylabel("Y")
    plt.legend()
    plt.savefig("1b.png")


def plotc(X, Y, theta_history, cost_history):
    n = 100
    left = -2
    right = 3
    theta0 = np.array([[(left + (right - left)*i/n)
                      for i in range(n)] for j in range(n)])
    theta1 = np.array([[(left + (right - left)*j/n)
                      for i in range(n)] for j in range(n)])
    j_theta = np.array([[cost(np.array([[theta1[i][j]], [theta0[i][j]]]), X, Y)
                       for i in range(n)] for j in range(n)])

    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection='3d')
    ax.set_xlabel("theta1")
    ax.set_ylabel("theta0")
    ax.set_zlabel("J(theta)")

    no_of_iter = len(theta_history)

    def init():
        ax.plot_surface(theta0, theta1, j_theta, cmap='viridis',
                        edgecolor='green', alpha=0.2, linewidth=0)
        return ax,

    def animate(i):
        theta = theta_history[i]
        cst = cost_history[i]
        z_points = np.array([i])
        ax.scatter3D(theta[0], theta[1], np.array(
            cost(theta, X, Y)), c=z_points, cmap='hsv')
        ax.text(theta[0][0], theta[1][0], cst, "(" + str("%.3f" % theta[0]
                [0]) + ", " + str("%.3f" % theta[1][0]) + ")", size=4, zorder=1, color='k')
        return ax,

    anim = FuncAnimation(fig, animate, init_func=init,
                         frames=no_of_iter, interval=200)
    # anim.save('1c.mp4', writer='ffmpeg')
    plt.show()


def plotd(X, Y, theta_history, cost_history):
    # Plot the contour
    n = 100
    left = -2
    right = 3
    fig1, ax1 = plt.subplots(figsize=(7, 7))
    theta0 = np.array([[(left + (right - left)*i/n)
                      for i in range(n)] for j in range(n)])
    theta1 = np.array([[(left + (right - left)*j/n)
                      for i in range(n)] for j in range(n)])
    j_theta = np.array([[cost(np.array([[theta1[i][j]], [theta0[i][j]]]), X, Y)
                       for i in range(n)] for j in range(n)])
    ax1.contour(theta0, theta1, j_theta, 100, cmap='jet')

    theta_0 = np.array([x[0][0] for x in theta_history])
    theta_1 = np.array([x[1][0] for x in theta_history])

    # Create animation
    line, = ax1.plot([], [], 'r', label='Gradient descent', lw=1.5)
    point, = ax1.plot([], [], '*', color='red', markersize=4)
    ax1.set_title("Eta = " + str(eta))
    ax1.set_xlabel("theta1")
    ax1.set_ylabel("theta0")

    def init():
        line.set_data([], [])
        point.set_data([], [])
        return line, point

    def animate(i):
        line.set_data(theta_0[:i], theta_1[:i])
        point.set_data(theta_0[i], theta_1[i])

        return line, point

    ax1.legend(loc=1)
    anim = FuncAnimation(fig1, animate, init_func=init,
                         frames=len(theta_0), interval=200, repeat=False)
    # anim.save('1e_' + str(eta) + '.gif', writer='ffmpeg')
    plt.show()


X, Y = load_data("data/q1/linearX.csv", "data/q1/linearY.csv")
X = normalized(X)

theta = np.zeros((X.shape[1], 1))

theta_history = [theta]
cost_history = [cost(theta, X, Y)]

t = 0
prev_cost = -1
curr_cost = cost_history[0]

while(abs(prev_cost - curr_cost) > 10e-20):
    theta = update(theta, eta, X, Y)

    prev_cost = curr_cost
    curr_cost = cost(theta, X, Y)

    theta_history.append(theta)
    cost_history.append(curr_cost)

    t = t + 1

print("No. of iterations:", t)
print("Theta:\n", theta)
print("Final Cost:", curr_cost)
# plotb(X, Y, theta)
# plotc(X, Y, theta_history, cost_history)
# plotd(X, Y, theta_history, cost_history)
# print_history(theta_history)
# print_history(cost_history)
