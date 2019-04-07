"""
Trains an MNIST classifier, and computes training sample influence on given test instances.
"""
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import model_util
from model_def import MNIST_Model
from data_util import MNIST_Data
from influence import Influence


def display_single_instance(data, fname='test_instance'):
    """Show single instance x, shape=(28, 28)."""

    image, label = data
    plt.imshow(image, cmap='gray')

    image_dir = 'images'
    os.makedirs(image_dir, exist_ok=True)
    plt.savefig(os.path.join(image_dir, fname + '.pdf'), format='pdf', bbox_inches='tight')


def display_multiple_instances(data, indices, fname=''):
    """Show multiple instances, each with shape=(28, 28)."""

    X, y = data
    fig, ax = plt.subplots(2, 5, figsize=(15, 5))
    target_ndx = 0

    for i in range(2):
        for j in range(5):
            image, label = X[target_ndx], np.argmax(y[target_ndx])
            ndx = indices[target_ndx]

            ax[i][j].set_axis_off()
            ax[i][j].imshow(image, cmap='gray')
            ax[i][j].set_title('[{}]: {}'.format(ndx, label))
            target_ndx += 1

    image_dir = 'images'
    os.makedirs(image_dir, exist_ok=True)
    plt.savefig(os.path.join(image_dir, fname + '.pdf'), format='pdf', bbox_inches='tight')


def main():

    tf.enable_eager_execution()

    mnist = MNIST_Data()
    model = MNIST_Model()

    train_ds, test_ds = mnist.get_dataset_iterators()

    model_util.restore(model)
    model_util.test(test_ds, model)

    inspector = Influence(train_loss_fn=model_util.loss_fn, test_loss_fn=model_util.loss_fn, model=model, data=mnist)
    up_inf = inspector.upweighting_influence(test_indices=[69], train_indices=np.arange(1000), force_refresh=False)
    indices = np.argsort(up_inf)

    k = 10
    neg_ndx = indices[:k]
    neg_inf = up_inf[neg_ndx]

    pos_ndx = indices[-k:][::-1]
    pos_inf = up_inf[pos_ndx]

    print('\nHarmful:')
    for ndx, inf in zip(neg_ndx, neg_inf):
        print('[{}] {}'.format(ndx, inf))

    print('\nHelpful:')
    for ndx, inf in zip(pos_ndx, pos_inf):
        print('[{}] {}'.format(ndx, inf))

    display_single_instance(mnist.get_test_instances(69))
    display_multiple_instances(mnist.get_train_instances(pos_ndx), pos_ndx, 'helpful')
    display_multiple_instances(mnist.get_train_instances(neg_ndx), neg_ndx, 'harmful')

if __name__ == '__main__':
    main()
