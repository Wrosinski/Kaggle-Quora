import pandas as pd
import numpy as np
import nltk
import multiprocessing
import difflib
import time
import gc
import xgboost as xgb
import category_encoders as ce
import itertools

from collections import Counter
from sklearn.metrics import log_loss
from sklearn.cross_validation import train_test_split



def labelcount_encode(df, cols):
    categorical_features = cols
    new_df = pd.DataFrame()
    for cat_feature in categorical_features:
        cat_feature_value_counts = df[cat_feature].value_counts()
        value_counts_list = cat_feature_value_counts.index.tolist()
        value_counts_range_rev = list(reversed(range(len(cat_feature_value_counts)))) # for ascending ordering
        value_counts_range = list(range(len(cat_feature_value_counts))) # for descending ordering
        labelcount_dict = dict(zip(value_counts_list, value_counts_range))
        new_df['{}_lc_encode'.format(cat_feature)] = df[cat_feature].map(labelcount_dict)
    return new_df

def count_encode(df, cols, normalize = False):
    categorical_features = cols
    new_df = pd.DataFrame()
    for i in categorical_features:
        new_df['{}_count_encode'.format(i)] = df[i].astype('object').replace(df[i].value_counts())
        if normalize:
            new_df['{}_count_encode'.format(i)] = new_df['{}_count_encode'.format(i)] / np.max(new_df['{}_count_encode'.format(i)])
    return new_df

def bin_numerical(df, cols, step):
    numerical_features = cols
    new_df = pd.DataFrame()
    for i in numerical_features:
        try:
            feature_range = np.arange(0, np.max(df[i]), step)
            new_df['{}_binned'.format(i)] = np.digitize(df[i], feature_range, right=True)
        except ValueError:
            df[i] = df[i].replace(np.inf, 999)
            feature_range = np.arange(0, np.max(df[i]), step)
            new_df['{}_binned'.format(i)] = np.digitize(df[i], feature_range, right=True)
    return new_df

def add_statistics(df, features_list):
    X = pd.DataFrame()
    X['sum_row_{}cols'.format(len(features_list))] = df[features_list].sum(axis = 1)
    X['mean_row{}cols'.format(len(features_list))] = df[features_list].mean(axis = 1)
    X['std_row{}cols'.format(len(features_list))] = df[features_list].std(axis = 1)
    X['max_row{}cols'.format(len(features_list))] = np.amax(df[features_list], axis = 1)
    print('Statistics of {} columns done.'.format(features_list))
    return X

def feature_combinations(df, features_list):
    X = pd.DataFrame()
    for comb in itertools.combinations(features_list, 2):
        feat = comb[0] + "_" + comb[1]
        X[feat] = df[comb[0]] * df[comb[1]]
    print('Interactions on {} columns done.'.format(features_list))
    return X

def group_featbyfeat(df, features_list, transformation):
    X = pd.DataFrame()
    for i in range(len(features_list) - 1):
        X['{}_by_{}_{}_list'.format(features_list[i], features_list[i+1], transformation)] = (df.groupby(features_list[i]))[features_list[i+1]].transform('{}'.format(transformation))
    print('Groupings of {} columns done.'.format(features_list))
    return X

def feature_comb_grouping(df, features_list, transformation):
    X = pd.DataFrame()
    for comb in itertools.combinations(features_list, 2):
        X['{}_by_{}_{}_combinations'.format(comb[0], comb[1], transformation)] = (df.groupby(comb[0]))[comb[1]].transform('{}'.format(transformation))
    print('Interactions on {} columns done.'.format(features_list))
    return X

def drop_duplicate_cols(df):
    dfc = df.iloc[:5000,:]
    dfc = dfc.T.drop_duplicates().T
    duplicate_cols = sorted(list(set(df.columns).difference(set(dfc.columns))))
    print('Dropping duplicate columns:', duplicate_cols)
    df.drop(duplicate_cols, axis = 1, inplace = True)
    print('Final shape:', df.shape)
    del dfc
    gc.collect()
    return df
