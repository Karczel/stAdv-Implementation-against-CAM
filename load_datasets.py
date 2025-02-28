from __future__ import division, absolute_import, print_function

import numpy as np
import tensorflow as tf


def load_mnist():
    # Load MNIST dataset directly
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Normalize the data (scale images to range [0, 1])
    # x_train, x_test = x_train / 255.0, x_test / 255.0
    # CNN Conv2D Layers format: (batch_size, height, width, channels)
    # By default, keras MNIST dataset is a grayscale (1 channal)
    # But MNIST dataset from tf.keras.datasets.mnist omitted it -> Shape of images: (10000, 28, 28)
    # So we reshape x to include the 'channels'
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    # but if RGB (3 channal), it  should be included by default
    # ex. Shape of images: (10000, 28, 28, 3)
    print("Shape of images:", x_test.shape)
    print("Range of values: from {} to {}".format(x_test.min(), x_test.max()))
    print("Shape of labels:", y_test.shape)
    print("Range of values: from {} to {}".format(y_test.min(), y_test.max()))
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_mnist()