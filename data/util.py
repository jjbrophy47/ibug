"""
Utility methods for preprocessing data.
"""
import sys
import time
import logging

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def get_logger(filename=''):
    """
    Return a logger object.
    """

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler(sys.stdout)
    log_handler = logging.FileHandler(filename, mode='w')
    formatter = logging.Formatter('%(message)s')

    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    log_handler.setLevel(logging.INFO)
    log_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(log_handler)

    return logger


def get_feature_names(column_transformer, logger=None):
    """
    Extract feature names from a ColumnTransformer object.
    """
    col_name = []

    # the last transformer is ColumnTransformer's 'remainder'
    for transformer_in_columns in column_transformer.transformers_[:-1]:
        if logger:
            logger.info('\n\ntransformer: ', transformer_in_columns[0])

        raw_col_name = list(transformer_in_columns[2])

        # if pipeline, get the last transformer
        if isinstance(transformer_in_columns[1], Pipeline):
            transformer = transformer_in_columns[1].steps[-1][1]

        else:
            transformer = transformer_in_columns[1]

        try:
            if isinstance(transformer, OneHotEncoder):
                names = list(transformer.get_feature_names_out(raw_col_name))

            elif isinstance(transformer, SimpleImputer) and transformer.add_indicator:
                missing_indicator_indices = transformer.indicator_.features_
                missing_indicators = [raw_col_name[idx] + '_missing_flag' for idx in missing_indicator_indices]
                names = raw_col_name + missing_indicators

            else:
                names = list(transformer.get_feature_names_out())

        except AttributeError:
            names = raw_col_name

        if logger:
            logger.info('{}'.format(names))

        col_name.extend(names)

    return col_name


def preprocess(train_df, test_df, features, val_df=None, logger=None, objective='classification'):
    """
    The bulk of the preprocessing for a dataset goes here.
    """

    # display datasets
    if logger:
        logger.info('\ntrain_df:\n{}\n{}'.format(train_df.head(5), train_df.shape))
        logger.info('test_df:\n{}\n{}'.format(test_df.head(5), test_df.shape))

    # count number of NAN values per column
    if logger:
        logger.info('')
        for c in train_df.columns:
            logger.info('[TRAIN] {}, no. missing: {:,}'.format(c, train_df[c].isna().sum()))
            logger.info('[TEST] {}, no. missing: {:,}'.format(c, test_df[c].isna().sum()))

    # display column info
    if logger:
        logger.info('\ntrain_df column info:')
        for c in train_df.columns:
            logger.info(f'{c}: {train_df[c].dtype}, {len(train_df[c].unique())}')

    # categorize attributes
    label = features['label']
    numeric = features['numeric']
    categorical = features['categorical']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))]
    )

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))]
    )

    # time transforms
    start = time.time()

    # perform one-hot encoding for all cat. attributes
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric),
            ('cat', categorical_transformer, categorical),
        ],
        sparse_threshold=0
    )

    # encode features
    train = preprocessor.fit_transform(train_df)
    test = preprocessor.transform(test_df)
    if val_df is not None:
        val = preprocessor.transform(val_df)

    # encode labels
    if objective == 'classification':
        le = LabelEncoder()
        train_label = le.fit_transform(train_df[label].values.ravel())
        test_label = le.transform(test_df[label].values.ravel())
        if val_df is not None:
            val_label = le.transform(val_df[label].values.ravel())

    # regression
    else:
        assert objective == 'regression'
        train_label = train_df[label].values.ravel()
        test_label = test_df[label].values.ravel()
        if val_df is not None:
            val_label = val_df[label].values.ravel()

    if logger:
        logger.info('transforming features...{:.3f}s'.format(time.time() - start))

    # get features
    feature_list = get_feature_names(preprocessor)
    assert len(feature_list) == train.shape[1] == test.shape[1]

    if val_df is not None:
        result = train, train_label, test, test_label, val, val_label, feature_list

    else:
        result = train, train_label, test, test_label, feature_list

    return result
