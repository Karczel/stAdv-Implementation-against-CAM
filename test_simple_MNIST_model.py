import tensorflow as tf
from keras.utils import to_categorical

from load_datasets import *


def test_mnist(x_test, y_test):
    # Restore the trained model
    restored_model = tf.keras.models.load_model('saved_models/simple_mnist.h5')
    y_test = to_categorical(y_test, 10)
    # Evaluate the restored model
    loss, accuracy = restored_model.evaluate(x_test, y_test)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")

test_mnist(x_test, y_test)
