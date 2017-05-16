import pandas as pd
import numpy as np
import nltk
from collections import Counter
from sklearn.metrics import log_loss
from scipy.optimize import minimize
import multiprocessing
import difflib
import time
import gc

import xgboost as xgb
from sklearn.cross_validation import train_test_split


def get_test():
    keras_q1 = np.load('../../data/transformed/keras_tokenizer/test_q1_transformed.npy')
    keras_q2 = np.load('../../data/transformed/keras_tokenizer/test_q2_transformed.npy')
    xgb_feats = pd.read_csv('../../data/features/the_1owl/owl_test.csv')
    abhishek_feats = pd.read_csv('../../data/features/abhishek/test_features.csv',
                              encoding = 'ISO-8859-1').iloc[:, 2:]
    text_feats = pd.read_csv('../../data/features/other_features/text_features_test.csv',
                            encoding = 'ISO-8859-1')
    img_feats = pd.read_csv('../../data/features/other_features/img_features_test.csv')
    srk_feats = pd.read_csv('../../data/features/srk/SRK_grams_features_test.csv')

    xgb_feats.drop(['z_len1', 'z_len2', 'z_word_len1', 'z_word_len2'], axis = 1, inplace = True)
    xgb_feats = xgb_feats.iloc[:, 5:]
    
    X_test2 = np.concatenate([keras_q1, keras_q2, xgb_feats, abhishek_feats, text_feats, img_feats], axis = 1)
    #X_test2 = np.concatenate([keras_q1, keras_q2, xgb_feats, abhishek_feats, text_feats], axis = 1)
    
    X_test2 = X_test2.astype('float32')
    X_test2 = pd.DataFrame(X_test2)
    print('Test data shape:', X_test2.shape)
    return X_test2

def predict_test(model_name):
    X_test = get_test()
    gbm = lgb.Booster(model_file='saved_models/LGBM/{}.txt'.format(model_name))
    test_preds = gbm.predict(lgb.Dataset(X_test))

    sub_src = '/media/w/1c392724-ecf3-4615-8f3c-79368ec36380/DS Projects/Kaggle/Quora/submissions/'
    sample_sub = pd.read_csv(sub_src + 'sample_submission.csv')
    sample_sub['is_duplicate'] = test_preds
    sample_sub.to_csv(sub_src + '{}.csv'.format(model_name), index = False)
    return

def oversample():
    print('Oversampling negative y according to anokas method')
    X_train, y_train = get_train()
    pos_train = X_train[X_train['is_duplicate'] == 1]
    neg_train = X_train[X_train['is_duplicate'] == 0]
    p = 0.165
    scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
    while scale > 1:
        neg_train = pd.concat([neg_train, neg_train])
        scale -=1
    neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
    X_train = pd.concat([pos_train, neg_train])
    y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()

    X_train = X_train.astype('float32')
    X_train.drop(['is_duplicate'], axis = 1, inplace = True)
    return X_train, y_train

def oversample2(X_train):
    print('Oversampling negative y according to SRK method')
    y_train = np.array(X_train["is_duplicate"])
    X_train.drop(['is_duplicate'], axis = 1, inplace = True)
    X_train_dup = X_train[y_train==1]
    X_train_non_dup = X_train[y_train==0]

    X_train = np.vstack([X_train_non_dup, X_train_dup, X_train_non_dup, X_train_non_dup])
    y_train = np.array([0]*X_train_non_dup.shape[0] + [1]*X_train_dup.shape[0] + [0]*X_train_non_dup.shape[0] + [0]*X_train_non_dup.shape[0])
    del X_train_dup
    del X_train_non_dup
    print("Mean target rate : ",y_train.mean())
    X_train = X_train.astype('float32')
    return X_train, y_train

def kappa(preds, y):
    score = []
    a = 0.165 / 0.37
    b = (1 - 0.165) / (1 - 0.37)
    for pp,yy in zip(preds, y.get_label()):
        score.append(a * yy * np.log (pp) + b * (1 - yy) * np.log(1-pp))
    score = -np.sum(score) / len(score)
    return 'kappa', score

def transform(x):
    a = 0.165 / 0.37
    b =  (1 - 0.165) / (1 - 0.37)
    xt = a * x / (a * x + b * (1 - x))
    return xt
