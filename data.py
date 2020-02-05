"""
Data object holds data
"""

from mnist.loader import MNIST
import numpy as np


class Data:
    def __init__(self):
        self.images = np.zeros(0)
        self.labels = np.zeros(0)
        self.size = 0

    def load(self, size, images, labels):
        """
        load MNIST data in. Expects output of MNIST(path).load_training() or .load_testing() as input
        :param size: size of input
        :param images: output of MNIST(path).load
        :param labels: output of MNIST(path).load
        :return:
        """
        self.size = size
        self.images = np.asarray(images, dtype=np.float16).reshape(size, 784) / 255  # divide to avoid huge weights
        self.labels = labels
        bias = np.ones((size, 1), dtype=np.float16)
        self.images = np.concatenate((self.images, bias), axis=1)  # concatenate matrix of 1s for bias