"""
Build a CNN classifier for MNIST. Uses Tensorflow: eager, dataset, keras.
"""
import tensorflow as tf


class MNIST_Model(tf.keras.Model):

    def __init__(self, device='cpu:0'):

        super(MNIST_Model, self).__init__()
        self.device = device
        self._input_shape = [-1, 28, 28, 1]   # inferred input, channels last

        # layer definitions
        self.conv1 = tf.layers.Conv2D(32, 5, padding='same', activation=tf.nn.relu)
        self.max_pool2d = tf.layers.MaxPooling2D((2, 2), (2, 2), padding='same')
        self.conv2 = tf.layers.Conv2D(64, 5, padding='same', activation=tf.nn.relu)
        self.fc1 = tf.layers.Dense(750, activation=tf.nn.relu)
        self.dropout = tf.layers.Dropout(0.5)
        self.fc2 = tf.layers.Dense(10)

    def call(self, x):
        x = tf.reshape(x, self._input_shape)
        x = self.max_pool2d(self.conv1(x))
        x = self.max_pool2d(self.conv2(x))
        x = tf.layers.flatten(x)
        x = self.dropout(self.fc1(x))
        return self.fc2(x)
