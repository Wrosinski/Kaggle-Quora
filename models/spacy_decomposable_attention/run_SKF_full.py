from __future__ import division, unicode_literals, print_function
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import warnings
warnings.filterwarnings('ignore')

import gc
import spacy
import plac
import time
import ujson as json
import numpy as np
import pandas as pd
import en_core_web_md
from tqdm import tqdm

from pathlib import Path
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import log_loss
try:
    import cPickle as pickle
except ImportError:
    import pickle

from spacy_hook import get_embeddings, get_word_ids
from spacy_hook import create_similarity_pipeline
from keras_decomposable_attention import build_model


def attention_foldrun(X, X2, y, name, Xte = None, Xte2 = None, start_fold = 0):
    
    skf = StratifiedKFold(n_splits = 10, random_state = 111, shuffle = True)
    if isinstance(X, pd.core.frame.DataFrame):
        X = X.values
    if isinstance(y, pd.core.frame.DataFrame):
        y = y.is_duplicate.values
    if isinstance(y, pd.core.frame.Series):
        y = y.values
    print('Running Decomposable Attention model with parameters:', settings)
    
    i = 1
    losses = []
    train_splits = []
    val_splits = []
    for tr_index, val_index in skf.split(X, y):
        train_splits.append(tr_index)
        val_splits.append(val_index)
        
    for i in range(start_fold, start_fold + 2):
        X_trq1, X_valq1 = X[train_splits[i]], X[val_splits[i]]
        X_trq2, X_valq2 = X2[train_splits[i]], X2[val_splits[i]]
        y_tr, y_val = y[train_splits[i]], y[val_splits[i]]
        y_tr = to_categorical(y_tr)
        y_val = to_categorical(y_val)
        t = time.time()
        
        print('Start training on fold: {}'.format(i))
        callbacks = [ModelCheckpoint('checks/decomposable_{}_10SKF_fold{}.h5'.format(i, name),
                                    monitor='val_loss', 
                                    verbose = 0, save_best_only = True),
                 EarlyStopping(monitor='val_loss', patience = 4, verbose = 1)]
        
        model = build_model(get_embeddings(nlp.vocab), shape, settings)
        model.fit([X_trq1, X_trq2], y_tr, validation_data=([X_valq1, X_valq2], y_val),
        nb_epoch=settings['nr_epoch'], batch_size=settings['batch_size'], callbacks = callbacks)
        val_pred = model.predict([X_valq1, X_valq2], batch_size = 64)
        score = log_loss(y_val, val_pred)
        losses.append(score)
        
        print('Predicting training set.')
        val_pred = pd.DataFrame(val_pred, index = val_splits[i])
        val_pred.columns = ['attention_feat1', 'attention_feat2']
        val_pred.to_pickle('OOF_preds/train_attentionpreds_fold{}.pkl'.format(i))
        print(val_pred.head())
        if Xte is not None:
            print('Predicting test set.')
            test_preds = model.predict([Xte, Xte2], batch_size = 64)
            test_preds = pd.DataFrame(test_preds)
            test_preds.columns = ['attention_feat1', 'attention_feat2']
            test_preds.to_pickle('OOF_preds/test_attentionpreds_fold{}.pkl'.format(i))
            del test_preds
            gc.collect()
            
        print('Final score for fold {} :'.format(i), score, '\n',
              'Time it took to train and predict on fold:', time.time() - t, '\n')
        del X_trq1, X_valq1, X_trq2, X_valq2, y_tr, y_val, val_pred
        gc.collect()
        i += 1
    print('Mean logloss for model in 10-folds SKF:', np.array(losses).mean(axis = 0))
    return


# In[ ]:

qsrc = '/media/w/1c392724-ecf3-4615-8f3c-79368ec36380/DS Projects/Kaggle/Quora/data/features/lemmatized_fullclean/'
qsrc2 = '/media/w/1c392724-ecf3-4615-8f3c-79368ec36380/DS Projects/Kaggle/Quora/data/features/NER/'
feats_src = '/media/w/1c392724-ecf3-4615-8f3c-79368ec36380/DS Projects/Kaggle/Quora/data/features/uncleaned/'
xgb_feats = pd.read_csv(feats_src + '/the_1owl/owl_train.csv')
y = xgb_feats.is_duplicate.values
nlp = en_core_web_md.load()
del xgb_feats
gc.collect()


# In[ ]:

settings = {
    'lr': 0.0005,
    'dropout': 0.2,
    'batch_size': 128,
    'nr_epoch': 200,
    'tree_truncate': True,
    'gru_encode': False,
    }

max_length = 128
nr_hidden = 256
shape = (max_length, nr_hidden, 2)
print('Shape setting:', shape)

q1n = np.load(qsrc2 + 'q1train_NER_128len.npy')
q2n = np.load(qsrc2 + 'q2train_NER_128len.npy')
q1nte = np.load(qsrc2 + 'q1test_NER_128len.npy')
q2nte = np.load(qsrc2 + 'q2test_NER_128len.npy')

attention_foldrun(q1n, q2n, y, 'NER128len_2ndrun', q1nte, q2nte, start_fold = 0)


# In[ ]:

settings = {
    'lr': 0.0005,
    'dropout': 0.2,
    'batch_size': 64,
    'nr_epoch': 100,
    'tree_truncate': True,
    'gru_encode': False,
    }

max_length = 170
nr_hidden = 256
shape = (max_length, nr_hidden, 2)
print('Shape setting:', shape)

q1 = np.load(qsrc + 'q1train_spacylemmat_fullclean_170len_treetrunc.npy')
q2 = np.load(qsrc + 'q2train_spacylemmat_fullclean_170len_treetrunc.npy')
q1te = np.load(qsrc2 + 'q1test_spacylemmat_fullclean_170len_treetrunc.npy')
q2te = np.load(qsrc2 + 'q2test_spacylemmat_fullclean_170len_treetrunc.npy')

attention_foldrun(q1, q2, y, 'CleanLemmat170len')



