# coding: utf8
import pickle, operator, os
import pandas as pd
import numpy as np
from formatting.csv_threads import csv_threads
from preprocessing.parse import preprocessing
from classifiers import deep_learning, machine_learning
from embeddings import tfidf, word2vec
from sklearn.metrics.pairwise import cosine_similarity
from ner.train import ner
# np.set_printoptions(threshold='nan')
# np.set_printoptions(suppress=True)



class classifier():

    def __init__(self,data_directory):
        # preprocess
        self.data_directory = data_directory

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
        #if not os.path.exists('./tmp/tfidf.pkl'):
        self.tfidf.train('./data/inputs/sentences.txt')
        self.tfidf.format_input()

        #the models
        ml = machine_learning.machine_learning()
        ml.prepare_data(test_size=0.20)
        results = {}
        for model_name, params_list in ml_models.items():
            results[model_name]={}
            for params in params_list:
                ml.build(model_name, params)
                ml.train()
                results[model_name][params] = ml.predict()
        if not os.path.exists('./tmp/results'):
            os.makedirs('./tmp/results')

        with open('./tmp/results/machine-learning.pkl', 'wb') as f:
            pickle.dump(results, f)

        print(results)



if __name__ == '__main__':

    ########
    #INPUTS#
    ########
    deep = False
    # dictionary that links machine learning models to their parameters
    ml_models = {}
    basic = [10**x for x in range(-3,3)]
    #ml_models['reglog_l1'] = [10]  # C
    ml_models['reglog_l2'] = [1]  # C
    #ml_models['reglog_sgd'] = 0.0001  # alpha
    #ml_models['naive_bayes'] = ''
    #ml_models['decision_tree'] = 'gini'  # entropy
    #ml_models['random_forest'] = [10,50,100,500]  # nb_estimator
    #ml_models['bagging_reglog_l1'] = 5  # nb_estimator
    #ml_models['bagging_reglog_l2'] = 5  # nb_estimator
    #ml_models['svm_linear'] = 1.0  # C
    #ml_models['knn'] = 5  # nb_neighbors

    data_directory = './data/SFR/messages_formated_cat.csv'
    data_directory_2 = './data/SFR/messages_motifs.csv'

    #Selection des data
    #On s'arrange pour ne garder que les deux colonnes sentence et label du csv
    # data = csv_threads(data_directory)
    # data.supprimer_cat(['custom_motif','sentence'])
    # data.changer_nom_colonne({'custom_motif':'label'})
    # cond = data.df.apply(lambda row:(pd.isnull(row['label'])==False), axis=1)
    # data.df = data.df[cond]
    # # data.csv_size()
    # data.csv_top_cat(14,'label')
    # data.dataframe_to_csv(data_directory_2)

    # data = csv_threads(data_directory_2)
    # data.df.groupby('label').size().size

    #Preprocessing
    preprocessing = preprocessing(data_directory_2)
    preprocessing.csv(word_label=True)

    # Classification
    classifier = classifier(data_directory_2)  # does preprocessing, with stemming,lemmatization,etc and put data in sentence.txt and label.txt
    classifier.train_ml()  # does the training
