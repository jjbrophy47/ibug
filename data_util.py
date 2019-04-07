"""
Custom MNIST data object to make retrieving specific parts of the data generic.
"""
import numpy as np
import tensorflow as tf


class MNIST_Data(object):

    def __init__(self):
        self.mnist = tf.keras.datasets.mnist
        self.x_train = None
        self.x_test = None

    def get_dataset_iterators(self, shuffle_size=10000, batch_size=30):

        (x_train, y_train), (x_test, y_test) = self.mnist.load_data()
        process = lambda x, y: (self._scale(x), tf.one_hot(y, 10))

        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_ds = train_ds.map(process).shuffle(shuffle_size).batch(batch_size)

        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        test_ds = test_ds.map(process).shuffle(shuffle_size).batch(batch_size)

        return train_ds, test_ds

    def get_train_instances(self, indices):
        if self.x_train is None:
            (x_train, y_train), (_, _) = self.mnist.load_data()
            self.x_train = tf.constant(x_train) / 255
            self.y_train = tf.one_hot(tf.constant(y_train), 10)

        x, y = tf.gather(self.x_train, indices), tf.gather(self.y_train, indices)
        return x, y

    def get_test_instances(self, indices):

        if self.x_test is None:
            (_, _), (x_test, y_test) = self.mnist.load_data()
            self.x_test = tf.constant(x_test) / 255
            self.y_test = tf.one_hot(tf.constant(y_test), 10)

        x, y = tf.gather(self.x_test, indices), tf.gather(self.y_test, indices)
        return x, y

    def get_all_train_indices(self):
        return np.arange(self.get_num_total_train_instances())

    def get_num_total_train_instances(self):
        (x_train, y_train), (x_test, y_test) = self.mnist.load_data()
        return len(x_train)

    def _scale(self, x, min_val=0.0, max_val=255.0):
        x = tf.to_float(x)
        return tf.div(tf.subtract(x, min_val), tf.subtract(max_val, min_val))
