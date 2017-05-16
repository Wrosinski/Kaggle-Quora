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
