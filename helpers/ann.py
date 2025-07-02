
import numpy as np


def initialize_nn(input_size, output_size):
    weights = kaiming_init(input_size, output_size)
    biases = np.zeros(output_size)

    return weights, biases

def feedforward(x, y_actual, weights, biases):
    
    nn_net = np.dot(x, weights) + biases
    y_pred = softmax(nn_net)

    # Compute categorical cross-entropy
    loss = -np.sum(y_actual * np.log(y_pred + 1e-9))

    gradient = y_pred - y_actual
    
    return y_pred, gradient, loss

def backpropagation(x, gradient, weights, biases, learning_rate=0.01):

    # np.outer multiplies x (size 15 or 8) to the gradient (size 10)
    # We compute here the gradient in respect to the weights
    dW = np.outer(x, gradient)
    
    # For the bias, the gradient will do
    db = gradient

    # Update weights and biases
    weights -= learning_rate * dW
    biases -= learning_rate * db

    return weights, biases

def kaiming_init(input_size, output_size):
    std = np.sqrt(2.0 / input_size)
    return np.random.normal(0.0, std, (input_size, output_size))

def softmax(z):
    z = z - np.max(z)
    e_z = np.exp(z)
    return e_z / np.sum(e_z)
