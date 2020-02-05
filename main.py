"""
This program contains a perceptron learning algorithm trained on the MNIST database.
Code for extracting MNIST images based on https://medium.com/@mannasiladittya/converting-mnist-data-in-idx-format-to-python-numpy-array-5cb9126f99f1
"""

import data
from perceptron import Perceptron
from mnist.loader import MNIST
import random
import datetime
import numpy as np

ETA = 0.2
ALPHA = 0.9
LOGFILE = 'eta01.txt'
CLASSES = 10
INPUT_SIZE = 2
HIDDEN_LAYERS = 1
HIDDEN_NODES = 2


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def toy():
    # create hidden layer
    hidden_weights = np.full((HIDDEN_NODES, INPUT_SIZE + 1), .1)
    # create output later
    output_weights = np.full(INPUT_SIZE + 1, .1)
    # create training images
    train_images = np.array([np.array([1, 0]), np.array([0, 1])])
    # Append 1s
    bias = np.ones((len(train_images), 1), dtype=np.uint8)
    train_images = np.concatenate((train_images, bias), axis=1)
    train_labels = [.9, -.3]
    # create testing images
    test_images = np.array([np.array([1, 1])])
    bias = np.ones((len(test_images), 1), dtype=np.uint8)
    test_images = np.concatenate((test_images, bias), axis=1)
    test_labels = [.8]
    # hidden sums
    hidden_sums = sigmoid(np.dot(train_images[0], np.transpose(hidden_weights)))
    print(hidden_sums)
    # output
    bias = np.ones(1, dtype=np.uint8)
    hidden_sums = np.concatenate((hidden_sums, bias), axis=0)
    print(f'hidden sums:{hidden_sums}')
    output = sigmoid(np.dot(hidden_sums, output_weights))
    print(f'output:{output}')
    d_output = output * (1 - output) * (train_labels[0] - output)
    print(f'd_output:{d_output}')
    d_hidden = hidden_sums * (1 - hidden_sums) * output_weights * d_output
    print(f'd_hidden:{d_hidden}')
    d_hidden_to_output = ETA * d_output * hidden_sums
    print(f'h_to_o:{d_hidden_to_output}')
    output_weights += d_hidden_to_output
    print(f'output weights:{output_weights}')
    for n in range(HIDDEN_NODES):
        d_input_to_hidden = ETA * d_hidden[n] * train_images[0]
        print(f'{n}th d_i_to_h: {d_input_to_hidden}')
        hidden_weights[n] += d_input_to_hidden
        print(f'{n}th hidden_weights: {hidden_weights[n]}')



if __name__ == '__main__':
    """
    mndata = MNIST('./images/')
    train_images, train_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()
    train_data = data.Data()
    train_data.load(60000, train_images, train_labels)
    test_data = data.Data()
    test_data.load(10000, test_images, test_labels)
    """
    toy()