from __future__ import absolute_import
import pickle
import numpy as np
import os, sys
from preprocessing.parse import preprocessing
import word2vec_paraphrase
import deep_learning_paraphrase
import tfidf_paraphrase
import machine_learning_paraphrase
# import tfidf_pd

deep = True
# dictionary that links machine learning models to their parameters
ml_models = {}
ml_models['reglog_l1'] = 1.0  # C
# ml_models['reglog_l2'] = 1.0  # C
# ml_models['reglog_sgd'] = 0.0001  # alpha
# ml_models['naive_bayes'] = ''
# ml_models['decision_tree'] = 'gini'  # entropy
# ml_models['random_forest'] = 5  # nb_estimator
# ml_models['bagging_reglog_l1'] = 5  # nb_estimator
# ml_models['bagging_reglog_l2'] = 5  # nb_estimator
# ml_models['svm_linear'] = 1.0  # C
# ml_models['knn'] = 5  # nb_neighbors

# inputs
data_directory = './data/MRPC/all.txt'
new_directory = './paraphrase_detection/data'

# preprocess the data within input/sentence.txt and input/label.txt
preprocessing = preprocessing(data_directory, new_directory)
preprocessing.paraphrase()

if deep is True:
    # compute word2vec and assign the right index to each word to make a proper numerical matrix to feed our model.
    word2vec = word2vec_paraphrase.word2vec_paraphrase(new_directory)
    word2vec.train(new_directory+'/input/sentences.txt', size=128, window=5, min_count=2)
    word2vec.format_input()

    # the models
    dl = deep_learning_paraphrase.deep_learning_paraphrase(new_directory)
    dl.prepare_data(test_size=0.20, max_len=23)
    dl.build_lstm_cnn(max_len=23, filter_length=3, nb_filter=64, pool_length=2, lstm_output_size=70)
    dl.train(batch_size=30, nb_epoch=1)
    dl.predict()
    dl.get_plots()
else:
    # compute tfidf and transform the sentences into tfidf vectors, with feature selection (best tfidf weights)
    tfidf = tfidf_paraphrase.tfidf_paraphrase(new_directory)
    tfidf.train(new_directory+'/input/sentences.txt')
    tfidf.format_input(feature_selection_threshold=0.85)

    # the models
    ml = machine_learning_paraphrase.machine_learning_paraphrase(new_directory)
    ml.prepare_data(test_size=0.20)
    for model_name, params in ml_models.items():
        ml.build(model_name, params)
        ml.train()
        ml.predict()
