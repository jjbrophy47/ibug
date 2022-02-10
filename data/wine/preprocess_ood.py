"""
Preprocess dataset to make it easier to load and work with.
"""
import os
import sys
import time
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')  # for utility
import util


def main(test_size=0.9, seed=1):

    # create logger
    logger = util.get_logger('log.txt')
    logger.info('timestamp: {}'.format(datetime.now()))

    # retrieve dataset
    start = time.time()
    df1 = pd.read_csv('winequality-white.csv', sep=';')
    df2 = pd.read_csv('winequality-red.csv', sep=';')
    df = pd.concat([df1, df2])
    print('\ntime to read in data...{:.3f}s'.format(time.time() - start))

    # split data into train and test
    train_df = df1
    test_df = df2
    val_df, test_df = train_test_split(test_df, test_size=test_size, random_state=seed)

    # get features
    columns = list(train_df.columns)

    # remove select columns
    remove_cols = []
    if len(remove_cols) > 0:
        train_df = train_df.drop(columns=remove_cols)
        test_df = test_df.drop(columns=remove_cols)
        columns = [x for x in columns if x not in remove_cols]

    # categorize attributes
    features = {}
    features['label'] = ['quality']
    features['numeric'] = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                           'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                           'pH', 'sulphates', 'alcohol']
    features['categorical'] = list(set(columns) - set(features['numeric']) - set(features['label']))

    X_train, y_train, X_test, y_test, X_val, y_val, feature = util.preprocess(train_df, test_df, features,
                                                                              val_df=val_df, logger=logger,
                                                                              objective='regression')

    data = {'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test,
            'X_val': X_val, 'y_val': y_val,
            'feature': feature}

    logger.info(f'train (head): {X_train[:5]}, {y_train[:5]}')
    logger.info(f'val (head): {X_val[:5]}, {y_val[:5]}')
    logger.info(f'test (head): {X_test[:5]}, {y_test[:5]}')
    logger.info(f'feature (head): {feature[:5]}')
    logger.info(f'X_train.shape: {X_train.shape}')
    logger.info(f'X_val.shape: {X_val.shape}')
    logger.info(f'X_test.shape: {X_test.shape}')
    logger.info(f'y_train.shape: {y_train.shape}, min., max.: {y_train.min()}, {y_train.max()}')
    logger.info(f'y_val.shape: {y_val.shape}, min., max.: {y_val.min()}, {y_val.max()}')
    logger.info(f'y_test.shape: {y_test.shape}, min., max.: {y_test.min()}, {y_test.max()}')

    # save to numpy format
    np.save(os.path.join('data_ood.npy'), data)


if __name__ == '__main__':
    main()
