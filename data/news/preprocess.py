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
    df = pd.read_csv('OnlineNewsPopularity.csv')
    logger.info('\ntime to read in data...{:.3f}s'.format(time.time() - start))

    # get features
    columns = list(df.columns)

    # remove select columns
    remove_cols = ['url', ' timedelta']
    if len(remove_cols) > 0:
        df = df.drop(columns=remove_cols)
        columns = [x for x in columns if x not in remove_cols]

    # categorize attributes
    features = {}
    features['label'] = [' shares']
    features['numeric'] = [' n_tokens_title', ' n_tokens_content', ' n_unique_tokens',
                           ' n_non_stop_words', ' n_non_stop_unique_tokens', ' num_hrefs', ' num_self_hrefs',
                           ' num_imgs', ' num_videos', ' average_token_length', ' num_keywords',
                           ' data_channel_is_lifestyle', ' data_channel_is_entertainment',
                           ' data_channel_is_bus', ' data_channel_is_socmed', ' data_channel_is_tech',
                           ' data_channel_is_world', ' kw_min_min', ' kw_max_min', ' kw_avg_min', ' kw_min_max',
                           ' kw_max_max', ' kw_avg_max', ' kw_min_avg', ' kw_max_avg', ' kw_avg_avg',
                           ' self_reference_min_shares', ' self_reference_max_shares', ' self_reference_avg_sharess',
                           ' weekday_is_monday', ' weekday_is_tuesday', ' weekday_is_wednesday',
                           ' weekday_is_thursday', ' weekday_is_friday', ' weekday_is_saturday',
                           ' weekday_is_sunday', ' is_weekend', ' LDA_00', ' LDA_01', ' LDA_02', ' LDA_03',
                           ' LDA_04', ' global_subjectivity', ' global_sentiment_polarity',
                           ' global_rate_positive_words', ' global_rate_negative_words', ' rate_positive_words',
                           ' rate_negative_words', ' avg_positive_polarity', ' min_positive_polarity',
                           ' max_positive_polarity', ' avg_negative_polarity', ' min_negative_polarity',
                           ' max_negative_polarity', ' title_subjectivity', ' title_sentiment_polarity',
                           ' abs_title_subjectivity', ' abs_title_sentiment_polarity']
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
