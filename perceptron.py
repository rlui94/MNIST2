"""
Perceptron object
"""

import numpy as np
import random


class Perceptron:
    def __init__(self, inputs):
        """
        Create a perceptron with randomized weights (including +1 for bias)
        :param inputs: number of inputs
        """
        # self.weights = np.random.rand(inputs + 1) - .5
        self.weights = np.full((inputs + 1), .1)
        self.value = 0

    def predict(self, inputs):
        """
        Return weighted sum of inputs. Does not squash.
        :param inputs: Input as numpy array
        :return: Weighted sum
        """
        self.value = self.sigmoid(np.dot(self.weights, inputs))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))