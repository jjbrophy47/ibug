"""
Preprocess dataset to make it easier to load and work with.

Preprocessing reference:
https://github.com/yromano/cqr/blob/master/datasets/datasets.py
"""
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')  # for utility
import util


def main(n_splits=10, random_state=1):

    # create logger
    logger = util.get_logger('log.txt')
    logger.info('timestamp: {}'.format(datetime.now()))

    col_names = ['Index', 'AGE', 'RACE', 'PCS42', 'MCS42', 'K6SUM42', 'UTILIZATION_reg',
                 'PERWT16F', 'REGION=1', 'REGION=2', 'REGION=3', 'REGION=4',
                 'SEX=1', 'SEX=2', 'MARRY=1', 'MARRY=2', 'MARRY=3', 'MARRY=4',
                 'MARRY=5', 'MARRY=6', 'MARRY=7', 'MARRY=8', 'MARRY=9', 'MARRY=10',
                 'FTSTU=-1', 'FTSTU=1', 'FTSTU=2', 'FTSTU=3', 'ACTDTY=1', 'ACTDTY=2',
                 'ACTDTY=3', 'ACTDTY=4', 'HONRDC=1', 'HONRDC=2', 'HONRDC=3', 'HONRDC=4',
                 'RTHLTH=-1', 'RTHLTH=1', 'RTHLTH=2', 'RTHLTH=3', 'RTHLTH=4', 'RTHLTH=5',
                 'MNHLTH=-1', 'MNHLTH=1', 'MNHLTH=2', 'MNHLTH=3', 'MNHLTH=4',
                 'MNHLTH=5', 'HIBPDX=-1', 'HIBPDX=1', 'HIBPDX=2', 'CHDDX=-1',
                 'CHDDX=1', 'CHDDX=2', 'ANGIDX=-1', 'ANGIDX=1', 'ANGIDX=2',
                 'MIDX=-1', 'MIDX=1', 'MIDX=2', 'OHRTDX=-1', 'OHRTDX=1', 'OHRTDX=2',
                 'STRKDX=-1', 'STRKDX=1', 'STRKDX=2', 'EMPHDX=-1', 'EMPHDX=1',
                 'EMPHDX=2', 'CHBRON=-1', 'CHBRON=1', 'CHBRON=2', 'CHOLDX=-1',
                 'CHOLDX=1', 'CHOLDX=2', 'CANCERDX=-1', 'CANCERDX=1', 'CANCERDX=2',
                 'DIABDX=-1', 'DIABDX=1', 'DIABDX=2', 'JTPAIN=-1', 'JTPAIN=1',
                 'JTPAIN=2', 'ARTHDX=-1', 'ARTHDX=1', 'ARTHDX=2', 'ARTHTYPE=-1',
                 'ARTHTYPE=1', 'ARTHTYPE=2', 'ARTHTYPE=3', 'ASTHDX=1', 'ASTHDX=2',
                 'ADHDADDX=-1', 'ADHDADDX=1', 'ADHDADDX=2', 'PREGNT=-1', 'PREGNT=1',
                 'PREGNT=2', 'WLKLIM=-1', 'WLKLIM=1', 'WLKLIM=2', 'ACTLIM=-1',
                 'ACTLIM=1', 'ACTLIM=2', 'SOCLIM=-1', 'SOCLIM=1', 'SOCLIM=2',
                 'COGLIM=-1', 'COGLIM=1', 'COGLIM=2', 'DFHEAR42=-1', 'DFHEAR42=1',
                 'DFHEAR42=2', 'DFSEE42=-1', 'DFSEE42=1', 'DFSEE42=2',
                 'ADSMOK42=-1', 'ADSMOK42=1', 'ADSMOK42=2', 'PHQ242=-1', 'PHQ242=0',
                 'PHQ242=1', 'PHQ242=2', 'PHQ242=3', 'PHQ242=4', 'PHQ242=5',
                 'PHQ242=6', 'EMPST=-1', 'EMPST=1', 'EMPST=2', 'EMPST=3', 'EMPST=4',
                 'POVCAT=1', 'POVCAT=2', 'POVCAT=3', 'POVCAT=4', 'POVCAT=5',
                 'INSCOV=1', 'INSCOV=2', 'INSCOV=3']

    # retrieve dataset
    start = time.time()
    df = pd.read_csv('get_meps_data/meps_21_reg.csv')
    logger.info('\ntime to read in data...{:.3f}s'.format(time.time() - start))

    # get features
    columns = list(df.columns)

    # remove select columns
    remove_cols = ['Unnamed: 0']
    if len(remove_cols) > 0:
        df = df.drop(columns=remove_cols)
        columns = [x for x in columns if x not in remove_cols]

    # categorize attributes
    features = {}
    features['label'] = ['UTILIZATION_reg']
    features['numeric'] = [c for c in columns if c not in ['UTILIZATION_reg']]
    features['categorical'] = list(set(columns) - set(features['numeric']) - set(features['label']))

    # split data into train and test
    fold = 1
    data = {}

    rs = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    for train_idxs, test_idxs in rs.split(df):
        logger.info(f'\nfold {fold}...')
        train_df = df.iloc[train_idxs]
        test_df = df.iloc[test_idxs]

        X_train, y_train, X_test, y_test, feature = util.preprocess(train_df, test_df, features,
                                                                    logger=logger if fold == 1 else None,
                                                                    objective='regression')
        data[fold] = {'X_train': X_train, 'y_train': y_train,
                      'X_test': X_test, 'y_test': y_test, 'feature': feature}
        fold += 1

    logger.info(f'\nfold {fold - 1} preview:')
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
