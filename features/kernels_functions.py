import numpy as np
import pandas as pd
import gensim
import re
import nltk
import json
import sys
import datetime
import operator
import matplotlib.pyplot as plt
import math
import csv
import timeit

from collections import Counter
from nltk.corpus import stopwords
from nltk import word_tokenize, ngrams
from gensim import corpora, models, similarities
from gensim.models.doc2vec import Doc2Vec
from gensim.models import doc2vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pylab import plot, show, subplot, specgram, imshow, savefig
from tqdm import tqdm


# https://www.kaggle.com/philschmidt/quora-question-pairs/quora-eda-model-selection-roc-pr-plots
def normalized_word_share(row):
        w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
        w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
        return 1.0 * len(w1 & w2)/(len(w1) + len(w2))
    
def modelselection_features(df2):
    df = df2.copy()
    df['normalized_word_share'] = df.apply(normalized_word_share, axis=1)
    return df


# https://www.kaggle.com/sudalairajkumar/quora-question-pairs/simple-exploration-notebook-quora-ques-pair
def feature_extraction(row):
    que1 = str(row['question1'])
    que2 = str(row['question2'])
    out_list = []
    # get unigram features #
    unigrams_que1 = [word for word in que1.lower().split() if word not in eng_stopwords]
    unigrams_que2 = [word for word in que2.lower().split() if word not in eng_stopwords]
    common_unigrams_len = len(set(unigrams_que1).intersection(set(unigrams_que2)))
    common_unigrams_ratio = float(common_unigrams_len) / max(len(set(unigrams_que1).union(set(unigrams_que2))),1)
    out_list.extend([common_unigrams_len, common_unigrams_ratio])

    # get bigram features #
    bigrams_que1 = [i for i in ngrams(unigrams_que1, 2)]
    bigrams_que2 = [i for i in ngrams(unigrams_que2, 2)]
    common_bigrams_len = len(set(bigrams_que1).intersection(set(bigrams_que2)))
    common_bigrams_ratio = float(common_bigrams_len) / max(len(set(bigrams_que1).union(set(bigrams_que2))),1)
    out_list.extend([common_bigrams_len, common_bigrams_ratio])

    # get trigram features #
    trigrams_que1 = [i for i in ngrams(unigrams_que1, 3)]
    trigrams_que2 = [i for i in ngrams(unigrams_que2, 3)]
    common_trigrams_len = len(set(trigrams_que1).intersection(set(trigrams_que2)))
    common_trigrams_ratio = float(common_trigrams_len) / max(len(set(trigrams_que1).union(set(trigrams_que2))),1)
    out_list.extend([common_trigrams_len, common_trigrams_ratio])
    return out_list


# https://www.kaggle.com/dasolmar/xgb-with-whq-jaccard
def get_weight(count, eps=10000, min_count=2):
    return 0 if count < min_count else 1 / (count + eps)

def add_word_count(x, df, word):
    x['q1_' + word] = df['question1'].apply(lambda x: (word in str(x).lower())*1)
    x['q2_' + word] = df['question2'].apply(lambda x: (word in str(x).lower())*1)
    x[word + '_both'] = x['q1_' + word] * x['q2_' + word]

def word_shares(row):
    q1_list = str(row['question1']).lower().split()
    q1 = set(q1_list)
    q1words = q1.difference(eng_stopwords)
    if len(q1words) == 0:
        return '0:0:0:0:0:0:0:0'

    q2_list = str(row['question2']).lower().split()
    q2 = set(q2_list)
    q2words = q2.difference(eng_stopwords)
    if len(q2words) == 0:
        return '0:0:0:0:0:0:0:0'

    words_hamming = sum(1 for i in zip(q1_list, q2_list) if i[0]==i[1])/max(len(q1_list), len(q2_list))

    q1stops = q1.intersection(eng_stopwords)
    q2stops = q2.intersection(eng_stopwords)

    q1_2gram = set([i for i in zip(q1_list, q1_list[1:])])
    q2_2gram = set([i for i in zip(q2_list, q2_list[1:])])

    shared_2gram = q1_2gram.intersection(q2_2gram)

    shared_words = q1words.intersection(q2words)
    shared_weights = [weights.get(w, 0) for w in shared_words]
    q1_weights = [weights.get(w, 0) for w in q1words]
    q2_weights = [weights.get(w, 0) for w in q2words]
    total_weights = q1_weights + q1_weights

    R1 = np.sum(shared_weights) / np.sum(total_weights) #tfidf share
    R2 = len(shared_words) / (len(q1words) + len(q2words) - len(shared_words)) #count share
    R31 = len(q1stops) / len(q1words) #stops in q1
    R32 = len(q2stops) / len(q2words) #stops in q2
    Rcosine_denominator = (np.sqrt(np.dot(q1_weights,q1_weights))*np.sqrt(np.dot(q2_weights,q2_weights)))
    Rcosine = np.dot(shared_weights, shared_weights)/Rcosine_denominator
    if len(q1_2gram) + len(q2_2gram) == 0:
        R2gram = 0
    else:
        R2gram = len(shared_2gram) / (len(q1_2gram) + len(q2_2gram))
    return '{}:{}:{}:{}:{}:{}:{}:{}'.format(R1, R2, len(shared_words), R31, R32, R2gram, Rcosine, words_hamming)

# https://www.kaggle.com/antriksh5235/doc2vec-starter
def clean_sentence(sent):
    regex = re.compile('([^\s\w]|_)+')
    sentence = regex.sub('', sent).lower()
    sentence = sentence.split(" ")
    for word in list(sentence):
        if word in eng_stopwords:
            sentence.remove(word)
    sentence = " ".join(sentence)
    return sentence

def cosine(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.dot(v1, v2) / (np.sqrt(np.sum(v1**2)) * np.sqrt(np.sum(v2**2)))

def concatenate(data):
    X_set1 = data['question1']
    X_set2 = data['question2']
    X = X_set1.append(X_set2, ignore_index=True)
    return X

class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield doc2vec.TaggedDocument(words=word_tokenize(doc),
                                         tags=[self.labels_list[idx]])
            
def get_dists_doc2vec(data, i):
    docvec1s = []
    docvec2s = []
    for i in tqdm(range(data.shape[0])):
        doc1 = word_tokenize(data.iloc[i, -2])
        doc2 = word_tokenize(data.iloc[i, -1])
        docvec1 = model1.infer_vector(doc1, alpha=start_alpha, steps=infer_epoch)
        docvec2 = model1.infer_vector(doc2, alpha=start_alpha, steps=infer_epoch)
        docvec1s.append(docvec1)
        docvec2s.append(docvec2)
    return docvec1s, docvec2s





src_train = 'df_train_NER.csv'
src_test = 'df_test_NER.csv'

df_train = pd.read_csv(src_train)
df_test = pd.read_csv(src_test)

eng_stopwords = set(stopwords.words('english'))
train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
words = (" ".join(train_qs)).lower().split()
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}
tic0=timeit.default_timer()
