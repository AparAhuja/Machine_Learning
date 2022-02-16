import numpy as np
import matplotlib.pyplot as plt


def load_data(fileX, fileY):
    x = np.genfromtxt(fileX)
    y = np.genfromtxt(fileY, dtype=str)
    for i in range(len(y)):
        if y[i] == 'Canada':
            y[i] = 1
        else:
            y[i] = 0
    y = y.reshape((y.shape[0], 1))
    return x, y

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


def plotb(class0, class1, theta):
    plt.scatter(class0[:, :1], class0[:, 1:2], 4, 'g', label='Alaska')
    plt.scatter(class1[:, :1], class1[:, 1:2], 4, 'b', label='Canada')
    left, right = plt.xlim()
    x = np.linspace(left, right)
    y = (- theta[0]*x - theta[2]) / theta[1]
    # plt.plot(x, y, 'r', label='Decision Boundry')
    plt.legend()
    plt.xlabel("Growth ring diameter in Fresh Water")
    plt.ylabel("Growth ring diameter in Marine Water")
    plt.savefig("4b.png")


def plotc(class0, class1, theta):
    plt.scatter(class0[:, :1], class0[:, 1:2], 4, 'g', label='Alaska')
    plt.scatter(class1[:, :1], class1[:, 1:2], 4, 'b', label='Canada')
    left, right = plt.xlim()
    x = np.linspace(left, right)
    y = (- theta[0]*x - theta[2]) / theta[1]
    plt.plot(x, y, 'r', label='Decision Boundry')
    plt.legend()
    plt.xlabel("Growth ring diameter in Fresh Water")
    plt.ylabel("Growth ring diameter in Marine Water")
    plt.savefig("4c.png")


def plote(class0, class1, theta):
    plt.scatter(class0[:, :1], class0[:, 1:2], 4, 'g', label ='Alaska')
    plt.scatter(class1[:, :1], class1[:, 1:2], 4, 'b', label='Canada')
    left, right = plt.xlim()
    x = np.linspace(left, right)
    y = (- theta[0]*x - theta[2]) / theta[1]
    plt.plot(x, y, 'r')
    n = 100
    left, right = plt.xlim()
    theta0 = np.array([[(left + (right - left)*i/n)
                      for i in range(n)] for j in range(n)])
    left, right = plt.ylim()
    theta1 = np.array([[(left + (right - left)*j/n)
                      for i in range(n)] for j in range(n)])
    j_theta = np.array([[quad_val(theta1[i][j], theta0[i][j])[0][0]
                       for i in range(n)] for j in range(n)])
    plt.contour(theta0, theta1, j_theta, [0])
    plt.xlabel("Growth ring diameter in Fresh Water")
    plt.ylabel("Growth ring diameter in Marine Water")
    plt.legend()
    plt.savefig("4e.png")


def gda_a(class0, class1):
    u0 = np.sum(class0, axis=0, keepdims=True)/class0.shape[0]
    u1 = np.sum(class1, axis=0, keepdims=True)/class1.shape[0]
    sigma = np.zeros((class0.shape[1], class0.shape[1]))
    for x in class0:
        sigma += np.dot((x - u0).T, x - u0)
    for x in class1:
        sigma += np.dot((x - u1).T, x - u1)
    sigma /= (class0.shape[0] + class1.shape[0])
    phi = class1.shape[0] / (class0.shape[0] + class1.shape[0])
    return u0, u1, sigma, phi


def gda_d(class0, class1):
    u0 = np.sum(class0, axis=0, keepdims=True)/class0.shape[0]
    u1 = np.sum(class1, axis=0, keepdims=True)/class1.shape[0]
    sigma0 = np.zeros((class0.shape[1], class0.shape[1]))
    sigma1 = np.zeros((class0.shape[1], class0.shape[1]))
    for x in class0:
        sigma0 += np.dot((x - u0).T, x - u0)
    for x in class1:
        sigma1 += np.dot((x - u1).T, x - u1)
    sigma0 /= class0.shape[0]
    sigma1 /= class1.shape[0]
    phi = class1.shape[0] / (class0.shape[0] + class1.shape[0])
    return u0, u1, sigma0, sigma1, phi


def print_history(l):
    for x in l:
        print(x)


X, Y = load_data("data/q4/q4x.dat", "data/q4/q4y.dat")
X = normalized(X)

class0 = []
class1 = []
for i in range(X.shape[0]):
    if Y[i][0] == '1':
        class1.append(X[i])
    else:
        class0.append(X[i])
class0 = np.array(class0)
class1 = np.array(class1)
u0, u1, sigma, phi = gda_a(class0, class1)
print("Part 1 - ")
print("mu0", u0, "mu1", u1, "sigma", sigma, "phi = " + str(phi), sep='\n')

u0, u1, sigma0, sigma1, phi = gda_d(class0, class1)
print("\n________________________________________________________\n")
print("Part 2 - ")
print("mu0", u0, "mu1", u1, "sigma0", sigma0,
      "sigma1", sigma1, "phi = " + str(phi), sep='\n')


def quad_val(x, y):
    X = np.array([[x], [y]])
    return np.dot(X.T, np.dot(np.linalg.inv(sigma1) - np.linalg.inv(sigma0), X))/2 - np.dot(np.dot(u1, np.linalg.inv(sigma1)) - np.dot(u0, np.linalg.inv(sigma0)), X) + np.dot(u1, np.dot(np.linalg.inv(sigma1), u1.T))/2 - np.dot(u0, np.dot(np.linalg.inv(sigma0), u0.T))/2 + np.log((1-phi)/phi*np.sqrt(np.linalg.det(sigma1)/np.linalg.det(sigma0)))


theta = 2*np.dot((u0 - u1), np.linalg.inv(sigma))
const = np.dot(np.dot(u1, np.linalg.inv(sigma)), u1.T) - \
    np.dot(np.dot(u0, np.linalg.inv(sigma)), u0.T) - 2*np.log(phi/(1-phi))
theta = np.append(theta, np.array(const))

# plotb(class0, class1, theta.T)
# plotc(class0, class1, theta.T)
# plote(class0, class1, theta.T)
