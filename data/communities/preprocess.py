"""
Preprocess dataset to make it easier to load and work with.

Preprocessing reference:
https://github.com/yromano/cqr/blob/master/datasets/datasets.py
"""
import os
import sys
import time
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')  # for utility
import util


def main(n_splits=10, random_state=1):

    # create logger
    logger = util.get_logger('log.txt')
    logger.info('timestamp: {}'.format(datetime.now()))

    columns = ['state', 'county', 'community', 'communityname', 'fold', 'population', 'householdsize',
               'racepctblack', 'racePctWhite', 'racePctAsian', 'racePctHisp', 'agePct12t21', 'agePct12t29',
               'agePct16t24', 'agePct65up', 'numbUrban', 'pctUrban', 'medIncome', 'pctWWage', 'pctWFarmSelf',
               'pctWInvInc', 'pctWSocSec', 'pctWPubAsst', 'pctWRetire', 'medFamInc', 'perCapInc',
               'whitePerCap', 'blackPerCap', 'indianPerCap', 'AsianPerCap', 'OtherPerCap', 'HispPerCap',
               'NumUnderPov', 'PctPopUnderPov', 'PctLess9thGrade', 'PctNotHSGrad', 'PctBSorMore',
               'PctUnemployed', 'PctEmploy', 'PctEmplManu', 'PctEmplProfServ', 'PctOccupManu',
               'PctOccupMgmtProf', 'MalePctDivorce', 'MalePctNevMarr', 'FemalePctDiv', 'TotalPctDiv',
               'PersPerFam', 'PctFam2Par', 'PctKids2Par', 'PctYoungKids2Par', 'PctTeen2Par',
               'PctWorkMomYoungKids', 'PctWorkMom', 'NumIlleg', 'PctIlleg', 'NumImmig', 'PctImmigRecent',
               'PctImmigRec5', 'PctImmigRec8', 'PctImmigRec10', 'PctRecentImmig', 'PctRecImmig5',
               'PctRecImmig8', 'PctRecImmig10', 'PctSpeakEnglOnly', 'PctNotSpeakEnglWell', 'PctLargHouseFam',
               'PctLargHouseOccup', 'PersPerOccupHous', 'PersPerOwnOccHous', 'PersPerRentOccHous',
               'PctPersOwnOccup', 'PctPersDenseHous', 'PctHousLess3BR', 'MedNumBR', 'HousVacant',
               'PctHousOccup', 'PctHousOwnOcc', 'PctVacantBoarded', 'PctVacMore6Mos', 'MedYrHousBuilt',
               'PctHousNoPhone', 'PctWOFullPlumb', 'OwnOccLowQuart', 'OwnOccMedVal', 'OwnOccHiQuart',
               'RentLowQ', 'RentMedian', 'RentHighQ', 'MedRent', 'MedRentPctHousInc', 'MedOwnCostPctInc',
               'MedOwnCostPctIncNoMtg', 'NumInShelters', 'NumStreet', 'PctForeignBorn', 'PctBornSameState',
               'PctSameHouse85', 'PctSameCity85', 'PctSameState85', 'LemasSwornFT', 'LemasSwFTPerPop',
               'LemasSwFTFieldOps', 'LemasSwFTFieldPerPop', 'LemasTotalReq', 'LemasTotReqPerPop',
               'PolicReqPerOffic', 'PolicPerPop', 'RacialMatchCommPol', 'PctPolicWhite', 'PctPolicBlack',
               'PctPolicHisp', 'PctPolicAsian', 'PctPolicMinor', 'OfficAssgnDrugUnits', 'NumKindsDrugsSeiz',
               'PolicAveOTWorked', 'LandArea', 'PopDens', 'PctUsePubTrans', 'PolicCars', 'PolicOperBudg',
               'LemasPctPolicOnPatr', 'LemasGangUnitDeploy', 'LemasPctOfficDrugUn', 'PolicBudgPerPop',
               'ViolentCrimesPerPop']

    # retrieve dataset
    start = time.time()
    df = pd.read_csv('communities.data', names=columns)
    print('\ntime to read in data...{:.3f}s'.format(time.time() - start))

    # impute
    df = df.replace('?', np.nan)
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer = imputer.fit(df[['OtherPerCap']])
    df[['OtherPerCap']] = imputer.transform(df[['OtherPerCap']])
    df = df.dropna(axis=1)

    # get features
    columns = list(df.columns)

    # remove select columns
    remove_cols = ['state', 'communityname', 'fold']
    if len(remove_cols) > 0:
        df = df.drop(columns=remove_cols)
        columns = [x for x in columns if x not in remove_cols]

    # categorize attributes
    features = {}
    features['label'] = ['ViolentCrimesPerPop']
    features['numeric'] = [c for c in columns if c not in remove_cols + features['label']]
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
