"""
This program contains a perceptron learning algorithm trained on the MNIST database.
Code for extracting MNIST images based on https://medium.com/@mannasiladittya/converting-mnist-data-in-idx-format-to-python-numpy-array-5cb9126f99f1
"""

import data
from perceptron import Perceptron
from mnist.loader import MNIST
from random import random
import datetime
import numpy as np

ETA = 0.1
ALPHA = 0.9
LOGFILE = 'eta01.txt'
CLASSES = 10
TOY_INPUT_SIZE = 2
HIDDEN_LAYERS = 1
HIDDEN_NODES = 2
INPUT_SIZE = 784
EPOCHS = 1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def toy():
    # create hidden layer
    hidden_weights = np.full((HIDDEN_NODES, TOY_INPUT_SIZE + 1), .1)
    # create output later
    output_weights = np.full(TOY_INPUT_SIZE + 1, .1)
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


def check_accuracy(data, hidden_weights, output_weights):
    correct = 0
    for i in range(data.size):
        # begin forward feed
        hidden_sums = sigmoid(np.dot(data.images[i], np.transpose(hidden_weights)))
        # Append 1 for bias in hidden layer
        bias = np.ones(1, dtype=np.uint8)
        hidden_sums_b = np.concatenate((hidden_sums, bias), axis=0)
        output = sigmoid(np.dot(hidden_sums_b, np.transpose(output_weights)))
        predict = np.argmax(output)
        if predict == data.labels[i]:
            correct += 1
    return correct/data.size


if __name__ == '__main__':
    # get MNIST data
    mndata = MNIST('./images/')
    train_images, train_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()
    train_data = data.Data()
    train_data.load(60000, train_images, train_labels)
    test_data = data.Data()
    test_data.load(10000, test_images, test_labels)
    # create hidden layer (weights from hidden to input)
    hidden_weights = np.random.rand(HIDDEN_NODES, INPUT_SIZE + 1) - .5
    hidden_moment = np.zeros((HIDDEN_NODES, INPUT_SIZE + 1))
    # create output layer (weights from output to hidden)
    output_weights = np.random.rand(CLASSES, HIDDEN_NODES + 1) - .5
    output_moment = np.zeros((CLASSES, HIDDEN_NODES + 1))
    # begin training
    train_acc = check_accuracy(train_data, hidden_weights, output_weights)
    test_acc = check_accuracy(test_data, hidden_weights, output_weights)
    accuracy = np.array([0, train_acc, test_acc])
    print(f'Initial accuracy: {train_acc} / {test_acc}')
    for e in range(EPOCHS):
        for i in range(3):
            # begin forward feed
            hidden_sums = sigmoid(np.dot(train_data.images[i], np.transpose(hidden_weights)))
            # Append 1 for bias in hidden layer
            bias = np.ones(1, dtype=np.uint8)
            hidden_sums_b = np.concatenate((hidden_sums, bias), axis=0)
            output = sigmoid(np.dot(hidden_sums_b, np.transpose(output_weights)))
            # set target values
            target_matrix = np.full(CLASSES, .1)
            target_matrix[train_data.labels[i]] = .9
            # find delta for output nodes
            d_output = np.multiply(np.multiply(output, (1 - output)), (target_matrix - output))
            # find delta for hidden nodes
            d_hidden = np.multiply(np.multiply(hidden_sums, (1 - hidden_sums)), np.dot(d_output, output_weights[:,:HIDDEN_NODES]))
            # find weight change for hidden to output
            d_hidden_to_output = np.multiply(ETA, np.outer(d_output, hidden_sums_b)) + np.multiply(ALPHA, output_moment)
            # store changes as momentum
            output_moment = d_hidden_to_output
            # adjust weights
            output_weights += d_hidden_to_output
            # find weight change for input to hidden
            d_input_to_hidden = np.multiply(ETA, np.outer(d_hidden, train_data.images[i])) + np.multiply(ALPHA, hidden_moment)
            # store changes as momentum
            hidden_moment = d_input_to_hidden
            # adjust weights
            hidden_weights += d_input_to_hidden
        train_acc = check_accuracy(train_data, hidden_weights, output_weights)
        test_acc = check_accuracy(test_data, hidden_weights, output_weights)
        print(f'Epoch {e + 1} Train/Test accuracy: {train_acc} / {test_acc}')
        np.append(accuracy, np.array([e + 1, train_acc, test_acc]))
    with open(LOGFILE, 'a') as file:
        file.write(str(accuracy))

