import nltk
import difflib
import time
import gc
import itertools
import multiprocessing
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split, StratifiedKFold


def predict_test(model_name):
    print('Predicting on test set.')
    X_test = get_test()
    gbm = xgb.Booster(model_file = 'saved_models/XGB/{}.txt'.format(model_name))
    test_preds = gbm.predict(xgb.DMatrix(X_test))

    sub_src = '/media/w/1c392724-ecf3-4615-8f3c-79368ec36380/DS Projects/Kaggle/Quora/submissions/'
    sample_sub = pd.read_csv(sub_src + 'sample_submission.csv')
    sample_sub['is_duplicate'] = test_preds
    sample_sub.is_duplicate = sample_sub.is_duplicate.apply(transform)
    sample_sub.to_csv(sub_src + '{}.csv'.format(model_name), index = False)
    return

def transform(x):
    a = 0.165 / 0.37
    b =  (1 - 0.165) / (1 - 0.37)
    xt = a * x / (a * x + b * (1 - x))
    return xt


def run_lgb(X_train, y_train, params, name):

    t = time.time()
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, stratify = y_train,
                                                    test_size = 0.2, random_state = 111)
    lgb_train = lgb.Dataset(X_tr, y_tr.is_duplicate.values)
    lgb_val = lgb.Dataset(X_val, y_val.is_duplicate.values)
    print('Start training with parameters:', params)
    gbm = lgb.train(params, lgb_train, num_boost_round = 100000, valid_sets = lgb_val,
                    early_stopping_rounds = 250, verbose_eval = 100)
    print('Start predicting...')
    val_pred = gbm.predict(X_val, num_iteration=gbm.best_iteration)
    score = log_loss(y_val, val_pred)
    print('Final score:', score, '\n', 'Time it took to train and predict:', time.time() - t)
    gbm.save_model('saved_models/LGBM/LGBM_{}_loss{:.5f}.txt'.format(name, score))
    return gbm

def run_xgb(X_train, y_train, params, name):

    t = time.time()
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, stratify = y_train,
                                                test_size = 0.2, random_state = 111)
    dtrain = xgb.DMatrix(X_tr, label = y_tr)
    dval = xgb.DMatrix(X_val, label = y_val)
    watchlist = [(dtrain, 'train'), (dval, 'valid')]
    print('Start training with parameters:', params)
    gbm = xgb.train(params, dtrain, 100000, watchlist, 
                    early_stopping_rounds = 250, verbose_eval = 100)
    print('Start predicting...')
    train_pred = gbm.predict(xgb.DMatrix(X_tr), ntree_limit=gbm.best_ntree_limit)
    val_pred = gbm.predict(xgb.DMatrix(X_val), ntree_limit=gbm.best_ntree_limit)
    score = log_loss(y_val, val_pred)
    print('Final score:', score, '\n', 'Time it took to train and predict:', time.time() - t)
    gbm.save_model('saved_models/XGB/XGB_{}_{:.5f}.txt'.format(name, score))
    return gbm
