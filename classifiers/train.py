# coding: utf8
import pickle, operator, os
import numpy as np
import formating
from preprocessing.parse import preprocessing
from classifiers import deep_learning, machine_learning
from embeddings import tfidf, word2vec
from sklearn.metrics.pairwise import cosine_similarity
from ner.train import ner
# np.set_printoptions(threshold='nan')
# np.set_printoptions(suppress=True)


#Parameters
data_directory = './data/SFR/messages_formated_cat.csv'

class classifier():

    def __init__(self, data_directory, word_label=True):
        # preprocess
        self.data_directory = data_directory
        self.preprocessing = preprocessing(data_directory)
        self.preprocessing.csv(word_label=word_label)

    def train_dl(self):
        # compute word2vec and assign the right index to each word to make a proper numerical matrix to feed our model.
        self.word2vec = word2vec.word2vec()
        if not os.path.exists('./tmp/word2vec'):
            self.word2vec.train('./data/inputs/sentences.txt', size=128)
        self.word2vec.format_input()

        # the models
        dl = deep_learning.deep_learning()
        dl.prepare_data(test_size=0.20, max_len=150)
        dl.build_cnn_lstm(max_len=150, filter_length=3, nb_filter=64, pool_length=2, lstm_output_size=70)
        dl.train(batch_size=30, nb_epoch=2)
        dl.predict()
        dl.get_plots()

    def train_ml(self):
        # compute tfidf and transform the sentences into tfidf vectors, with feature selection (best tfidf weights)
        self.tfidf = tfidf.tfidf()
        if not os.path.exists('./tmp/tfidf.pkl'):
            self.tfidf.train('./data/inputs/sentences.txt')
        self.tfidf.format_input()

        #the models
        ml = machine_learning.machine_learning()
        ml.prepare_data(test_size=0.20)
        for model_name, params in ml_models.items():
            ml.build(model_name, params)
            ml.train()
            ml.predict()


if __name__ == '__main__':

    deep = False
    # dictionary that links machine learning models to their parameters
    ml_models = {}
    # ml_models['reglog_l1'] = 1.0  # C
    ml_models['reglog_l2'] = 1.0  # C
    # ml_models['reglog_sgd'] = 0.0001  # alpha
    # ml_models['naive_bayes'] = ''
    # ml_models['decision_tree'] = 'gini'  # entropy
    # ml_models['random_forest'] = 50  # nb_estimator
    # ml_models['bagging_reglog_l1'] = 5  # nb_estimator
    # ml_models['bagging_reglog_l2'] = 5  # nb_estimator
    # ml_models['svm_linear'] = 1.0  # C
    # ml_models['knn'] = 5  # nb_neighbors

    data_directory = './data/SFR/messages_formated_cat.csv'
    classifier = classifier(data_directory)  # does preprocessing
    classifier.train_dl()  # does the training
