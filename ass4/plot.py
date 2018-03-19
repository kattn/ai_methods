from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


# Takes a 1x2 weight matrix and a 1x2 datapoint
# Returns a activation number
def sigmoid(w,x):
    return 1/(1+np.exp(-x*w.T))

# Takes a 1x2 weight matrix and a 1x2 datapoint
# Returns a activation number
def sigmoid_der(w,x):
    return sigmoid(w,x)*(1-sigmoid(w,x))

# Takes a 1x2 weight matrix
# Returns a loss
def loss(w):
    x1 = np.matrix("1 0");
    x2 = np.matrix("0 1");
    x3 = np.matrix("1 1");
    return (sigmoid(w, x1) - 1)**2 + (sigmoid(w, x2))**2 + (sigmoid(w, x3) - 1)**2

# Takes a 1x2 weight matrix
# Returns a 1x2 gradient matrix
def loss_der(w):
    x1 = np.matrix("1 0");
    x2 = np.matrix("0 1");
    x3 = np.matrix("1 1");
    return np.matrix(str(2*(sigmoid(w,x1)-1)*sigmoid_der(w,x1) + 2*(sigmoid(w,x3)-1)*sigmoid_der(w,x3)) + " " +
            str(2*sigmoid(w,x2)*sigmoid_der(w,x2) + 2*(sigmoid(w,x3)-1)*sigmoid_der(w,x3)))

# Takes a 1x2 weight matrix and a learning rate eta
# Returns a new 1x2 weight matrix
def update_rule(w, eta):
    return (w - loss_der(w)*eta)

# Takes a 1x2 weight matrix, a learning rate eta and number of iterations
# Returns a new 1x2 weight matrix
def grad_decent(w_init, eta, num_iterations, history=False):
    if(history):
        w = [ w_init.copy() ]
        for i in range(num_iterations):
            w.append(update_rule(w[-1], eta))
        return w

    w_current = w_init.copy()
    for i in range(num_iterations):
        w_current = update_rule(w_current, eta)
    return w_current


# L1
# From plot I see that [-6, 3] are shit initial weights
shit_weights = np.matrix("-6 3")

init_weights = shit_weights
etas = [0.0001, 0.01, 0.1, 1, 10, 100]
iterations = 100
results = {}

for eta in etas:
    print("Learning rate:", eta)
    results[eta] = grad_decent(init_weights, eta, iterations, history=True)
    for it in range(0, iterations+5, 5):
        print("Iterations: {:-2d} Weights: {:30} Loss: {}".format(it, str(results[eta][it]), str(loss(results[eta][it]).A[0][0])))



# ===================
# PLOTTING
# ===================
fig = plt.figure(1)


# ===================
# SUBPLOT 1 - Loss function
# ===================
ax = fig.add_subplot(2,2,1, projection='3d')
ax.set_title("Loss function")

# Make data.
X = np.arange(-6, 6.25, 0.25)
Y = np.arange(-6, 6.25, 0.25)
X, Y = np.meshgrid(X, Y)
zs = np.array([loss(np.matrix(str(x) + "," + str(y))) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.rainbow,
                       linewidth=0, antialiased=False)

# Customize the axis.
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_xlabel("w1")
ax.set_ylabel("w2")

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

# ===================
# SUBPLOT 2 - Loss for each learningreate over iterations
# ===================

ax = fig.add_subplot(2,2,2)
ax.set_xlabel("Iterations")
ax.set_ylabel("Loss")

for eta in etas:
    ax.plot([loss(result).A[0][0] for result in results[eta]])
ax.legend([str(eta) for eta in etas], title="Learning rates", bbox_to_anchor=(1,1))

fig.tight_layout()
plt.show()
