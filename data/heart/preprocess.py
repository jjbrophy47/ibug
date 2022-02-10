"""
Generates a 9x9 heart dataset.
"""
import os
import sys
import math
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')  # for utility
import util


def gen_parameters(rng, noise=0.01):
    """
    Function for generating mean and variance.
    """
    mean = rng.uniform(size=(9, 9))

    figure = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 4, 1, 1, 1, 4, 1, 1],
        [1, 4, 0, 4, 1, 4, 0, 4, 1],
        [4, 0, 0, 0, 4, 0, 0, 0, 4],
        [1, 4, 0, 0, 0, 0, 0, 4, 1],
        [1, 1, 4, 0, 0, 0, 4, 1, 1],
        [1, 1, 1, 4, 0, 4, 1, 1, 1],
        [1, 1, 1, 1, 4, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1]
    ])

    var = figure * noise

    return mean, var


def generate_training_data(rng, n_samples, mean, var, num_cat=9):
    """
    Function for generating train and validation sets.
    """
    train = []
    target = []

    for i in range(num_cat):

        for j in range(num_cat):
            if var[i, j] == 0:
                continue

            for _ in range(n_samples):
                train.append([i, j])
                target.append(rng.normal(mean[i, j], math.sqrt(var[i, j])))

    train = np.asarray(train)
    target = np.asarray(target)

    return train, target


def main(random_state=1, n_samples=1000):

    # create logger
    logger = util.get_logger('log.txt')
    logger.info('timestamp: {}'.format(datetime.now()))

    # psuedo-random number generator
    rng = np.random.default_rng(random_state)

    # generate parameters and plot variance
    mean, var = gen_parameters(rng)
    fig, ax = plt.subplots()
    sns.heatmap(var, cmap="Reds", vmax=0.05, ax=ax)
    ax.set_title('True Data Uncertainty (variance)')
    ax.set_xlabel('x0')
    ax.set_ylabel('x1')
    plt.savefig("heart_data_gt.pdf")

    # generate train dataset
    X_train, y_train = generate_training_data(rng, n_samples, mean, var)

    # generate test, consisting of all possible feature combinations
    num_cat = 9
    X_test = np.asarray([[i, j] for i in range(num_cat) for j in range(num_cat)])
    y_test = np.asarray([rng.normal(mean[i, j], math.sqrt(var[i, j])) for i in range(num_cat) for j in range(num_cat)])

    train_df = pd.DataFrame(np.hstack([X_train, y_train.reshape(-1, 1)]), columns=['x0', 'x1', 'y'])
    test_df = pd.DataFrame(np.hstack([X_test, y_test.reshape(-1, 1)]), columns=['x0', 'x1', 'y'])

    # get features
    columns = list(train_df.columns)

    # categorize attributes
    features = {}
    features['label'] = ['y']
    features['numeric'] = []
    features['categorical'] = list(set(columns) - set(features['numeric']) - set(features['label']))

    X_train, y_train, X_test, y_test, feature = util.preprocess(train_df, test_df, features,
                                                                logger=logger, objective='regression')

    feature = ['x{}'.format(x) for x in range(X_train.shape[1])]

    data = {'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test, 'feature': feature}

    logger.info(f'train (head): {X_train[:5]}, {y_train[:5]}')
    logger.info(f'test (head): {X_test[:5]}, {y_test[:5]}')
    logger.info(f'feature (head): {feature[:5]}')
    logger.info(f'X_train.shape: {X_train.shape}')
    logger.info(f'X_test.shape: {X_test.shape}')
    logger.info(f'y_train.shape: {y_train.shape}, min., max.: {y_train.min()}, {y_train.max()}')
    logger.info(f'y_test.shape: {y_test.shape}, min., max.: {y_test.min()}, {y_test.max()}')

    # save to numpy format
    np.save(os.path.join('data.npy'), data)


if __name__ == '__main__':
    main()
