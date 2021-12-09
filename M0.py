"""
Neural Networks - Deep Learning
MRI Brain Tumor Classifier ('pituitary' | 'no tumor' | 'meningioma' | 'glioma')
Author: Dimitrios Spanos Email: dimitrioss@ece.auth.gr
"""
import numpy as np
from scipy import signal
from M0_auxiliary import my_Layer, my_Activation

# -------------------------------
#             LAYERS
# -------------------------------


class my_Linear(my_Layer):
    def __init__(self, input_size, output_size, initialization=None):
        if initialization is None:
            self.weights = np.random.randn(output_size, input_size)
            self.bias = np.random.randn(output_size, 1)
        elif initialization == 'He':
            self.weights = np.random.uniform(low=-np.sqrt(6 / output_size),
                                            high=np.sqrt(6 / output_size),
                                            size=(output_size, input_size),)
            self.bias = np.random.randn(output_size, 1)

    # u = w^T x + b
    def forward(self, input):

        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):

        weights_gradient = np.dot(output_gradient, self.input.T)
        # update the weights
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return np.dot(self.weights.T, output_gradient)


class my_Conv2d(my_Layer):
    def __init__(self, input_shape, kernel_size, output_channels, initialization=None):
        input_depth, input_height, input_width = input_shape
        self.depth = output_channels
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (output_channels, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (output_channels, input_depth, kernel_size, kernel_size)

        self.kernels = np.random.randn(*self.kernels_shape)
        if initialization == 'He':
            for i in range(output_channels):
                if i==0:
                    # bottom sobel
                    self.kernels[i][0] = np.array([[-1,-2,-1],
                                                  [0, 0, 0],
                                                  [1, 2, 1]], np.float32)
                elif i==1:
                    # outline
                    self.kernels[i][0] = np.array([[-1, -1, -1],
                                                   [-1, +8, -1],
                                                   [-1, -1, -1]], np.float32)
                else:
                    # He Uniform
                    self.kernels[i][0] = np.random.uniform(low=-np.sqrt(6 / kernel_size),
                                                           high=np.sqrt(6 / kernel_size),
                                                           size=(kernel_size,kernel_size),)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        return self.output

    def backward(self, output_gradient, learning_rate):

        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")

        # update the weights
        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient

class my_Flatten(my_Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        return np.reshape(input, self.output_shape)

    def backward(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)


# -------------------------------
#       ACTIVATION FUNCTIONS
# -------------------------------

class my_Sigmoid(my_Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)


class my_Tanh(my_Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)


class my_Softmax(my_Layer): # used as the model's final activation function

    def __init__(self):
        self.forward_in = None
        self.forward_out = None
        self.backward_in = None
        self.backward_out = None

    def forward(self, x):
        self.forward_in = x
        self.forward_out = (
                np.exp(self.forward_in - np.max(self.forward_in)) / # avoid numerical overflow with "-np.max(x)"
                np.sum(np.exp(self.forward_in - np.max(self.forward_in)))
        )
        return self.forward_out

    def backward(self, x, learning_rate):
        self.backward_in = x
        if self.backward_in is None:
            self.backward_out = None
            return self.backward_out

        # Ensure that everything is in the right shape
        softmax = np.reshape(self.forward_out, (1, -1))
        grad = np.reshape(self.backward_in, (1, -1))

        d_softmax = (
                softmax * np.identity(softmax.size)
                - softmax.transpose() @ softmax)
        self.backward_out = (grad @ d_softmax).ravel() # ravel prevents problems with matrix multiplications
        return self.backward_out.reshape(-1, 1)

# -------------------------------
#             LOSSES
# -------------------------------


def my_cross_entropy(y, y_pred, epsilon=1e-12):

    # clip values to [10^(−12), 1 − 10^(−12)] range to avoid numerical issues
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    N = y_pred.shape[0]
    return -np.sum(y * np.log(y_pred+1e-9)) / N # +1e-9 is to avoid log(0)


def my_d_cross_entropy(y, y_pred):

    def softmax(x):
        exps = np.exp(x - x.max())
        return exps / np.sum(exps)

    y = np.argmax(y)
    grad = softmax(y_pred)
    grad[y] -= 1 # I decrease the gradient of the correct choice
    return grad

def my_MSE(y_true, y_pred):
    return np.mean(np.power(y_true -y_pred, 2))

def my_d_MSE(y_true, y_pred):
    d_MSE = 2 * (y_pred - y_true) / np.size(y_true)
    return d_MSE