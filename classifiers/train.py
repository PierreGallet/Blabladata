# coding: utf8
import pickle, operator, os, sys
import numpy as np
import formatting
from preprocessing.parse import preprocessing
from classifiers import deep_learning, machine_learning
from embeddings import tfidf, word2vec
from sklearn.metrics.pairwise import cosine_similarity
from ner.train import ner
# np.set_printoptions(threshold='nan')
# np.set_printoptions(suppress=True)



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
            print('word2vec not found, it will need to be trained with input data.')
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
            print('tfidf not found, it will need to be trained with input data.')
            self.tfidf.train('./data/inputs/sentences.txt')
        self.tfidf.format_input()

        #the models
        ml = machine_learning.machine_learning()
        ml.prepare_data(test_size=0.20)
        for model_name, params_list in ml_models.items():
            results[model_name] = {}
            for params in params_list:
                ml.build(model_name, params)
                ml.train()
                results[model_name][params] = ml.predict()


if __name__ == '__main__':

    ########
    #INPUTS#
    ########
    deep = False
    # dictionary that links machine learning models to their parameters
    ml_models = {}
    ml_models['reglog_l1'] = [0.5, 1.0, 5.0]  # C
    ml_models['reglog_l2'] = [0.5, 1.0, 5.0]  # C
    ml_models['reglog_sgd'] = [0.0001, 0.001, 0.01] # alpha
    ml_models['naive_bayes'] = [0.5, 1.0, 5.0]
    ml_models['decision_tree'] = ['gini', 'entropy']  # entropy
    ml_models['random_forest'] = [5, 10, 20]  # nb_estimator
    ml_models['bagging_reglog_l1'] = [5, 10, 20]  # nb_estimator (C is 1 by default)
    ml_models['bagging_reglog_l2'] = [5, 10, 20]  # nb_estimator (C is 1 by default)
    ml_models['svm_linear'] = [0.5, 1.0, 5.0]  # C
    ml_models['knn'] = [5, 10, 20]  # nb_neighbors

    data_directory = './data/SFR/messages_formated_cat.csv'


    results = {}
    # classifier = classifier(sys.argv[1])  # does preprocessing
    classifier = classifier(data_directory)  # does preprocessing
    classifier.train_ml()  # does the training
    print json.dumps(results, indent=4)
