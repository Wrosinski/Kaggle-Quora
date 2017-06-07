import os
import time
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split, StratifiedKFold


def lgb_foldrun(X, y, params, name):
    skf = StratifiedKFold(n_splits = 10, random_state = 111, shuffle = True)
    if isinstance(X, pd.core.frame.DataFrame):
        X = X.values
    if isinstance(y, pd.core.frame.DataFrame):
        y = y.is_duplicate.values
    if isinstance(y, pd.core.frame.Series):
        y = y.values
    print('Running LGBM model with parameters:', params)
        
    i = 1
    losses = []
    oof_train = np.zeros((X.shape[0]))
    os.makedirs('saved_models/LGBM/SKF/{}'.format(name), exist_ok = True)
    for tr_index, val_index in skf.split(X, y):
        X_tr, X_val = X[tr_index], X[val_index]
        y_tr, y_val = y[tr_index], y[val_index]
        t = time.time()
        
        lgb_train = lgb.Dataset(X_tr, y_tr)
        lgb_val = lgb.Dataset(X_val, y_val)
        print('Start training on fold: {}'.format(i))
        gbm = lgb.train(params, lgb_train, num_boost_round = 100000, valid_sets = lgb_val,
                        early_stopping_rounds = 200, verbose_eval = 100)
        print('Start predicting...')
        val_pred = gbm.predict(X_val, num_iteration=gbm.best_iteration)
        oof_train[val_index] = val_pred
        score = log_loss(y_val, val_pred)
        losses.append(score)
        print('Final score for fold {} :'.format(i), score, '\n',
              'Time it took to train and predict on fold:', time.time() - t, '\n')
        gbm.save_model('saved_models/LGBM/SKF/{}/LGBM_10SKF_loss{:.5f}_fold{}.txt'.format(name, score, i))
        i += 1
    np.save('OOF_preds/train/{}'.format(name), oof_train)
    print('Mean logloss for model in 10-folds SKF:', np.array(losses).mean(axis = 0))
    return


def xgb_foldrun(X, y, params, name):
    skf = StratifiedKFold(n_splits = 10, random_state = 111, shuffle = True)
    if isinstance(X, pd.core.frame.DataFrame):
        X = X.values
    if isinstance(y, pd.core.frame.DataFrame):
        y = y.is_duplicate.values
    if isinstance(y, pd.core.frame.Series):
        y = y.values
    print('Running XGB model with parameters:', params)
    
    i = 1
    losses = []
    oof_train = np.zeros((X.shape[0]))
    os.makedirs('saved_models/XGB/SKF/{}'.format(name), exist_ok = True)
    for tr_index, val_index in skf.split(X, y):
        X_tr, X_val = X[tr_index], X[val_index]
        y_tr, y_val = y[tr_index], y[val_index]
        t = time.time()
        
        dtrain = xgb.DMatrix(X_tr, label = y_tr)
        dval = xgb.DMatrix(X_val, label = y_val)
        watchlist = [(dtrain, 'train'), (dval, 'valid')]
        print('Start training on fold: {}'.format(i))
        gbm = xgb.train(params, dtrain, 100000, watchlist, 
                        early_stopping_rounds = 200, verbose_eval = 100)
        print('Start predicting...')
        val_pred = gbm.predict(xgb.DMatrix(X_val), ntree_limit=gbm.best_ntree_limit)
        oof_train[val_index] = val_pred
        score = log_loss(y_val, val_pred)
        losses.append(score)
        print('Final score for fold {} :'.format(i), score, '\n',
              'Time it took to train and predict on fold:', time.time() - t, '\n')
        gbm.save_model('saved_models/XGB/SKF/{}/XGB_10SKF_loss{:.5f}_fold{}.txt'.format(name, score, i))
        i += 1
    np.save('OOF_preds/train/{}'.format(name), oof_train)
    print('Mean logloss for model in 10-folds SKF:', np.array(losses).mean(axis = 0))
    return




def lgb_foldrun_test(X, y, X_test, params, name, save = True):
    skf = StratifiedKFold(n_splits = 10, random_state = 111, shuffle = True)
    if isinstance(X, pd.core.frame.DataFrame):
        X = X.values
    if isinstance(y, pd.core.frame.DataFrame):
        y = y.is_duplicate.values
    if isinstance(y, pd.core.frame.Series):
        y = y.values
    print('Running LGBM model with parameters:', params)
        
    i = 0
    losses = []
    oof_train = np.zeros((X.shape[0]))
    oof_test = np.zeros((10, 2345796))
    os.makedirs('saved_models/LGBM/SKF/{}'.format(name), exist_ok = True)
    for tr_index, val_index in skf.split(X, y):
        X_tr, X_val = X[tr_index], X[val_index]
        y_tr, y_val = y[tr_index], y[val_index]
        t = time.time()
        
        lgb_train = lgb.Dataset(X_tr, y_tr)
        lgb_val = lgb.Dataset(X_val, y_val)
        print('Start training on fold: {}'.format(i))
        gbm = lgb.train(params, lgb_train, num_boost_round = 100000, valid_sets = lgb_val,
                        early_stopping_rounds = 200, verbose_eval = 100)
        print('Start predicting...')
        val_pred = gbm.predict(X_val, num_iteration=gbm.best_iteration)
        oof_train[val_index] = val_pred
        score = log_loss(y_val, val_pred)
        losses.append(score)
        if X_test is not None:
            test_preds = gbm.predict(X_test, num_iteration=gbm.best_iteration)
            oof_test[i, :] = test_preds
        print('Final score for fold {} :'.format(i), score, '\n',
              'Time it took to train and predict on fold:', time.time() - t, '\n')
        gbm.save_model('saved_models/LGBM/SKF/{}/LGBM_10SKF_loss{:.5f}_fold{}.txt'.format(name, score, i))
        i += 1
    print('Mean logloss for model in 10-folds SKF:', np.array(losses).mean(axis = 0))
    oof_train = pd.DataFrame(oof_train)
    oof_train.columns = ['{}_prob'.format(name)]
    oof_test = oof_test.mean(axis = 0)
    oof_test = pd.DataFrame(oof_test)
    oof_test.columns = ['{}_prob'.format(name)]
    if save:
        oof_train.to_pickle('OOF_preds/train/train_preds_{}.pkl'.format(name))
        oof_test.to_pickle('OOF_preds/test/test_preds_{}.pkl'.format(name))
    return oof_train, oof_test

def xgb_foldrun_test(X, y, X_test, params, name, save = True):
    skf = StratifiedKFold(n_splits = 10, random_state = 111, shuffle = True)
    if isinstance(X, pd.core.frame.DataFrame):
        X = X.values
    if isinstance(y, pd.core.frame.DataFrame):
        y = y.is_duplicate.values
    if isinstance(y, pd.core.frame.Series):
        y = y.values
    print('Running XGB model with parameters:', params)
    
    i = 0
    losses = []
    oof_train = np.zeros((X.shape[0]))
    oof_test = np.zeros((10, 2345796))
    os.makedirs('saved_models/XGB/SKF/{}'.format(name), exist_ok = True)
    for tr_index, val_index in skf.split(X, y):
        X_tr, X_val = X[tr_index], X[val_index]
        y_tr, y_val = y[tr_index], y[val_index]
        t = time.time()
        
        dtrain = xgb.DMatrix(X_tr, label = y_tr)
        dval = xgb.DMatrix(X_val, label = y_val)
        watchlist = [(dtrain, 'train'), (dval, 'valid')]
        print('Start training on fold: {}'.format(i))
        gbm = xgb.train(params, dtrain, 100000, watchlist, 
                        early_stopping_rounds = 200, verbose_eval = 100)
        print('Start predicting...')
        val_pred = gbm.predict(xgb.DMatrix(X_val), ntree_limit=gbm.best_ntree_limit)
        oof_train[val_index] = val_pred
        score = log_loss(y_val, val_pred)
        losses.append(score)
        if X_test is not None:
            test_preds = gbm.predict(X_test, ntree_limit=gbm.best_ntree_limit)
            oof_test[i, :] = test_preds
        print('Final score for fold {} :'.format(i), score, '\n',
              'Time it took to train and predict on fold:', time.time() - t, '\n')
        gbm.save_model('saved_models/XGB/SKF/{}/XGB_10SKF_loss{:.5f}_fold{}.txt'.format(name, score, i))
        i += 1
    print('Mean logloss for model in 10-folds SKF:', np.array(losses).mean(axis = 0))
    oof_train = pd.DataFrame(oof_train)
    oof_train.columns = ['{}_prob'.format(name)]
    oof_test = oof_test.mean(axis = 0)
    oof_test = pd.DataFrame(oof_test)
    oof_test.columns = ['{}_prob'.format(name)]
    if save:
        oof_train.to_pickle('OOF_preds/train/train_preds_{}.pkl'.format(name))
        oof_test.to_pickle('OOF_preds/test/test_preds_{}.pkl'.format(name))
    return oof_train, oof_test
