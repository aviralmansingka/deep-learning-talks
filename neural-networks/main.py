import numpy as np
import sklearn.datasets
import sklearn.linear_model
import matplotlib.pyplot as plt


np.random.seed(0)
X, y = sklearn.datasets.make_moons(200, noise=0.20)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
plt.show()

def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()


clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X, y)

plot_decision_boundary(lambda x: clf.predict(x))

num_examples = len(X) # training set size
nn_input_dim = 2 # input layer dimensionality
nn_output_dim = 2 # output layer dimensionality

epsilon = 0.01 # learning rate for gradient descent
reg_lambda = 0.01

class Mul:
    @staticmethod
    def fwd(w, x):
        return np.dot(x, w)

    @staticmethod
    def back(dL, w, x):
        dW = np.dot(x.T, dL)
        dX = np.dot(dL, w.T)
        return dW, dX


class Add:
    @staticmethod
    def fwd(wx, b):
        return wx + b

    @staticmethod
    def back(dL, wx, b):
        dMul = np.ones_like(wx) * dL
        dB = np.dot(np.ones((1, dL.shape[0])), dL)
        return dB, dMul


class Tanh:
    @staticmethod
    def fwd(z):
        return np.tanh(z)

    @staticmethod
    def back(dL, z):
        fwd = Tanh.fwd(z)
        return dL * (1 - fwd**2)


class Softmax:
    @staticmethod
    def fwd(z):
        exp_score = np.exp(z)
        return exp_score / np.sum(exp_score, axis=1, keepdims=True)

    @staticmethod
    def back(y_hat, y):
        return 


def forward(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # TODO

    mul = Mul.fwd(x, W1)
    add = Add.fwd(mul, b1)
    act = Tanh.fwd(add, 

def loss(model):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    probs = forward(model, X)

    # likelyhood
    # numpy magic

    # regularization

    # divide by number of examples


def predict(model, x):
    # why multiple weights
    pass


def init_model(nn_hdim):
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))

    return {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2,
    }

def build_model(nn_hdim, num_passes=20000, print_loss=False):

    # Initialize the parameters to random values. We need to learn these.
    model = init_model(nn_hdim)

    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

    # Gradient descent. For each batch...
    for i in range(0, num_passes):

        # forward

        # Backpropagation

        # Add regularization terms (b1 and b2 don't have regularization terms)

        # Gradient descent parameter update

        # Assign new parameters to the model

        if print_loss and i % 1000 == 0:
          print("Completed iteration {}".format(i))

    return model


# Build a model with a 3-dimensional hidden layer
model = build_model(3, print_loss=True)

# Plot the decision boundary
plot_decision_boundary(lambda x: predict(model, x))
