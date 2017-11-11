import numpy as np


import sklearn.datasets
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


num_examples = len(X) # training set size
nn_input_dim = 2 # input layer dimensionality
nn_output_dim = 2 # output layer dimensionality

epsilon = 0.01 # learning rate for gradient descent
reg_lambda = 0.01

class Mul:
    @staticmethod
    def fwd(w, x):
        return np.dot(x,w)

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
        return dMul, dB

class Tanh:
    @staticmethod
    def fwd(z):
        return np.tanh(z)

    @staticmethod
    def back(dL, z):
        return (1 - np.tanh(z)**2) * dL

class Softmax:
    @staticmethod
    def fwd(z):
        exp_result = np.exp(z)
        return exp_result / np.sum(exp_result, axis=1, keepdims=True)

    @staticmethod
    def back(y_hat, y):
        y_hat[range(num_examples), y.flatten().astype(int)] -= 1
        return y_hat


def forward(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

    wx1 = Mul.fwd(W1, x)
    z1 = Add.fwd(wx1, b1)
    a1 = Tanh.fwd(z1)

    wx2 = Mul.fwd(W2, a1)
    z2 = Add.fwd(wx2, b2)
    a2 = Softmax.fwd(z2)

    return a2

def loss(model):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    probs = forward(model, X)

    # likelyhood
    # numpy magic
    correct_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(correct_logprobs)

    # regularization
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return data_loss / num_examples


def predict(model, x):
    # why multiple weights
    probs = forward(model, x)
    return np.argmax(probs, axis=1)


def build_model(nn_hdim, num_passes=20000, print_loss=False):

    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))

    # This is what we return at the end
    model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    # Gradient descent. For each batch...
    for i in range(0, num_passes):

        W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

        # forward
        wx1 = Mul.fwd(W1, X)
        z1 = Add.fwd(wx1, b1)
        a1 = Tanh.fwd(z1)

        wx2 = Mul.fwd(W2, a1)
        z2 = Add.fwd(wx2, b2)
        a2 = Softmax.fwd(z2)

        probs = a2

        # Backpropagation
        delta3 = Softmax.back(probs, y)

        dMul, db2 = Add.back(delta3, wx2, b2)
        dW2, dAct = Mul.back(dMul, W2, a1)

        delta2 = Tanh.back(dAct, z1)

        dMul, db1 = Add.back(delta2, wx1, b1)
        dW1, _ = Mul.back(delta2, W1, X)

        # Add regularization terms (b1 and b2 don't have regularization terms)
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1

        # Gradient descent parameter update
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2

        # Assign new parameters to the model
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        if print_loss and i % 1000 == 0:
          print("Completed iteration {}".format(i))

    return model


# Build a model with a 3-dimensional hidden layer
model = build_model(3, print_loss=True)

# Plot the decision boundary
plot_decision_boundary(lambda x: predict(model, x))
