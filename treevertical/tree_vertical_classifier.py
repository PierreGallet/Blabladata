# coding: utf-8
from __future__ import division
import pickle, operator, os
import pandas as pd
import numpy as np
import sys
from formatting.csv_threads import csv_threads
from preprocessing.parse import preprocessing
from copy import copy
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
import csv
import time, operator
import os
import sys
import json
from create_tree import create_tree
from preprocessing.parse import preprocessing
from embeddings import tfidf, word2vec
from classifiers import machine_learning
from create_data import list_node
import sklearn.linear_model as linear
import sklearn.svm as svm
import sklearn.tree as tree
import sklearn.ensemble as ens
import sklearn.neighbors as neighbors
import sklearn.naive_bayes as nb
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split, StratifiedKFold
import math
import random

np.random.seed(332)

class tree_vertical_classifier():

    def __init__(self,tree,rawdata,dataformated):
        self.tree = tree
        self.rawdata = rawdata
        self.df = pd.read_csv(rawdata, sep=';')
        self.dataformated = dataformated
        self.path_sentences = './data/inputs/tfidf/sentences.npy'
        self.path_labels = './data/inputs/tfidf/labels.npy'

    def formatting(self):
        data = csv_threads(self.rawdata)
        data.supprimer_cat(['phrase','intent'])
        data.changer_nom_colonne({'intent':'label','phrase':'sentence'})
        print(data.df)
        data.dataframe_to_csv(dataformated)

    def sep_motifs(self,column):
        print(self.df[column])
        labels = self.df[column]
        labels = labels.apply(lambda labe: labe.split('-'))
        for index,value_serie in labels.iteritems():
            for key,value in enumerate(value_serie):
                self.df.ix[index,column+'_'+str(key+1)] = value

    def vectorize_and_split(self,test_size = 0.20):
        # compute tfidf and transform the sentences into tfidf vectors, with feature selection (best tfidf weights)
        self.tfidf = tfidf.tfidf()
        #if not os.path.exists('./tmp/tfidf.pkl'):
        self.tfidf.train('./data/inputs/sentences.txt')
        self.tfidf.format_input()

        with open(self.path_sentences, 'rb') as sentences_npy:
            with open(self.path_labels, 'rb') as labels_npy:
                self.sentences = np.array(np.load(sentences_npy))
                self.labels = np.array(np.load(labels_npy))

        context = self.df['context']
        print('colonne',context)
        #print(context)
        self.context = context.as_matrix()
        print('self context',self.context)

        #print(X_train.shape, y_train.shape)
        #print(len(X_train), 'train sequences')
        #print(len(X_val), 'validation sequences')

    def train_node(self,context_node,test_size=0.2):
        print(context_node)
        if context_node=='':
            context_node = np.nan
        else:
            context_node = int(context_node)
        print(self.context[:, ])
        cond =(self.context[:, ] == context_node)
        print('cond',cond)
        X = self.sentences[cond,:]
        y = self.labels[cond,:]

        n = X.shape[0]
        ntrain = int(math.floor(X.shape[0]*0.8))

        print(n,ntrain)
        ech =random.sample(range(0,n),ntrain)
        ech_comp = [item for item in range(0,n) if item not in ech]

        X_train = X[ech,:]
        y_train = np.ravel(y[ech,])
        X_val = X[ech_comp]
        y_val = np.ravel(y[ech_comp,])

        #X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

        print(y_train,y_val)
        # the models
        results = {}
        print("\n\n\n------- Pour "+str(context_node)+"--------")
        for model_name, params_list in ml_models.items():
            results[model_name]={}
            for params in params_list:
                self.build(model_name, params)
                self.train(X_train,y_train)
                results[model_name][params] = self.predict(X_val,y_val)
        if not os.path.exists('./treevertical/results'):
            os.makedirs('./treevertical/results')

        with open("./treevertical/results/"+str(context_node)+".pkl", 'wb') as f:
            pickle.dump(results, f)
        print(results)

    def train_tree(self):
        liste_nodes = list_node(self.tree,[],'0',True)
        print(liste_nodes)
        for ele in liste_nodes:
            print(ele)
            context=ele[0]
            self.train_node(context,0.2)

    def build(self, model_name, params):
        print('...Build model...')
        self.model_name = model_name
        self.params = params
        if model_name == 'reglog_l1':
            self.model = linear.LogisticRegression(penalty='l1', C=params)
        elif model_name == 'reglog_l2':
            self.model = linear.LogisticRegression(penalty='l2', C=params)
        elif model_name == 'reglog_sgd':
            self.model = linear.SGDClassifier(loss="log", penalty='elasticnet', alpha=params)
        elif model_name == 'naive_bayes':
            self.model = nb.BernoulliNB(alpha=params)
        elif model_name == 'decision_tree':
            self.model = tree.DecisionTreeClassifier(criterion=params)
        elif model_name == 'random_forest':
            self.model = ens.RandomForestClassifier(n_estimators=params,max_features='sqrt',bootstrap=False)
        elif model_name == 'bagging_reglog_l1':
            self.model = ens.BaggingClassifier(base_estimator=linear.LogisticRegression(penalty='l1', C=0.5), n_estimators=params)
        elif model_name == 'bagging_reglog_l2':
            self.model = ens.BaggingClassifier(base_estimator=linear.LogisticRegression(penalty='l2', C=0.5), n_estimators=params)
        elif model_name == 'svm_linear':
            self.model = svm.LinearSVC(penalty='l2', C=params)
        elif model_name == 'knn':
            self.model = neighbors.KNeighborsClassifier(n_neighbors=params, metric='minkowski', weights='distance')

    def train(self,X_train,y_train):
        print('...Train...')
        start_time = time.time()
        self.model.fit(X_train,y_train)
        self.average_training_time = (time.time() - start_time)


        # if os.path.exists('./tmp/models_saved/'+self.model_name+'?p='+str(self.params)+'.pkl'):
        #     os.remove('./tmp/models_saved/'+self.model_name+'?p='+str(self.params)+'.pkl')

        # print('...Saving model...')
        # with open('./tmp/models_saved/'+self.model_name+'?p='+str(self.params)+'.pkl', 'wb') as f:
        #     pickle.dump(self.model, f)
        # print('...Model Saved...') # pourquoi aussi long de saver le modèle?

    def predict(self,X_val,y_val):
        with open('./tmp/models_saved/classes.json', 'rb') as f:
            classes = json.load(f)
            self.target_names = [labels for key, labels in classes.items()]

        for i in range(len(self.target_names)):
            self.target_names[i] = self.target_names[i]


        self.pred = self.model.predict(X_val)
        self.accuracy = accuracy_score(y_val, self.pred)
        #self.confusion_matrix = np.array(confusion_matrix(self.y_val, self.pred), dtype=float)
        #self.classification_report = classification_report(self.y_val, self.pred, target_names=self.target_names)
        #print(self.pred)
        print('\n\n## FINISHED ##')
        print('\nresult for the ' + self.model_name + ' on validation set:')
        #print('accuracy:', self.accuracy, '\nconfusion matrix:\n', self.confusion_matrix, '\naverage_training_time:', self.average_training_time, '\nclassification report', self.classification_report)
        print('accuracy:',self.accuracy,"average_training_time:",self.average_training_time)
        return [self.accuracy,self.average_training_time]



if __name__ == '__main__':

    rawdata = './treevertical/data/rawdata.csv'
    dataformated = './treevertical/data/dataformated.csv'

    test=create_tree()
    test.add_node_2([["intent a",'mess a'],["intent b", "mess b"],["intent c", "mess c"],["intent d", "mess d"]])
    test.initialize_tree([1,2])
    # test.add_branches([3,4,2],[1])
    # test.add_branches([1,3],[1,3])

    tree = test.tree

    deep = False
    # dictionary that links machine learning models to their parameters
    ml_models = {}
    #basic = [10**x for x in range(-3,3)]
    #ml_models['reglog_l1'] = [10]  # C
    ml_models['reglog_l2'] = [1]  # C


    #
    test = tree_vertical_classifier(tree,rawdata,dataformated)
    test.formatting()


    #Preprocessing
    preprocessing = preprocessing(dataformated)
    preprocessing.csv(word_label=False)

    test.vectorize_and_split()
    test.train_tree()
