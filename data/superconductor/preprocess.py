"""
Preprocess dataset to make it easier to load and work with.
"""
import os
import sys
import time
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')  # for utility
import util


def main(n_splits=10, random_state=1):

    # create logger
    logger = util.get_logger('log.txt')
    logger.info('timestamp: {}'.format(datetime.now()))

    # retrieve dataset
    start = time.time()
    df = pd.read_csv('superconduct/train.csv')
    logger.info('\ntime to read in data...{:.3f}s'.format(time.time() - start))

    # get features
    columns = list(df.columns)

    # remove select columns
    remove_cols = []
    if len(remove_cols) > 0:
        df = df.drop(columns=remove_cols)
        columns = [x for x in columns if x not in remove_cols]

    # categorize attributes
    features = {}
    features['label'] = ['critical_temp']
    features['numeric'] = ['number_of_elements', 'mean_atomic_mass', 'wtd_mean_atomic_mass',
                           'gmean_atomic_mass', 'wtd_gmean_atomic_mass', 'entropy_atomic_mass',
                           'wtd_entropy_atomic_mass', 'range_atomic_mass', 'wtd_range_atomic_mass', 'std_atomic_mass',
                           'wtd_std_atomic_mass', 'mean_fie', 'wtd_mean_fie', 'gmean_fie', 'wtd_gmean_fie',
                           'entropy_fie', 'wtd_entropy_fie', 'range_fie', 'wtd_range_fie', 'std_fie', 'wtd_std_fie',
                           'mean_atomic_radius', 'wtd_mean_atomic_radius', 'gmean_atomic_radius',
                           'wtd_gmean_atomic_radius', 'entropy_atomic_radius', 'wtd_entropy_atomic_radius',
                           'range_atomic_radius', 'wtd_range_atomic_radius', 'std_atomic_radius',
                           'wtd_std_atomic_radius', 'mean_Density', 'wtd_mean_Density', 'gmean_Density',
                           'wtd_gmean_Density', 'entropy_Density', 'wtd_entropy_Density', 'range_Density',
                           'wtd_range_Density', 'std_Density', 'wtd_std_Density', 'mean_ElectronAffinity',
                           'wtd_mean_ElectronAffinity', 'gmean_ElectronAffinity', 'wtd_gmean_ElectronAffinity',
                           'entropy_ElectronAffinity', 'wtd_entropy_ElectronAffinity', 'range_ElectronAffinity',
                           'wtd_range_ElectronAffinity', 'std_ElectronAffinity', 'wtd_std_ElectronAffinity',
                           'mean_FusionHeat', 'wtd_mean_FusionHeat', 'gmean_FusionHeat', 'wtd_gmean_FusionHeat',
                           'entropy_FusionHeat', 'wtd_entropy_FusionHeat', 'range_FusionHeat', 'wtd_range_FusionHeat',
                           'std_FusionHeat', 'wtd_std_FusionHeat', 'mean_ThermalConductivity',
                           'wtd_mean_ThermalConductivity', 'gmean_ThermalConductivity',
                           'wtd_gmean_ThermalConductivity', 'entropy_ThermalConductivity',
                           'wtd_entropy_ThermalConductivity', 'range_ThermalConductivity',
                           'wtd_range_ThermalConductivity', 'std_ThermalConductivity', 'wtd_std_ThermalConductivity',
                           'mean_Valence', 'wtd_mean_Valence', 'gmean_Valence', 'wtd_gmean_Valence',
                           'entropy_Valence', 'wtd_entropy_Valence', 'range_Valence', 'wtd_range_Valence',
                           'std_Valence', 'wtd_std_Valence', 'critical_temp']
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

    logger.info(f'\nFold {fold - 1} preview:')
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
