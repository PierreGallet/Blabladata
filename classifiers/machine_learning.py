# coding: utf-8
""" Train sklearn's most relevant models """
from __future__ import print_function, division
#import matplotlib.pyplot as plt
import numpy as np
import time, pickle, os, json
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


class machine_learning():


    def __init__(self):
        self.path_sentences = './data/inputs/tfidf/sentences.npy'
        self.path_labels = './data/inputs/tfidf/labels.npy'


    def prepare_data(self, test_size=0.20):
        with open(self.path_sentences, 'rb') as sentences_npy:
            with open(self.path_labels, 'rb') as labels_npy:
                sentences = np.array(np.load(sentences_npy))
                labels = np.array(np.load(labels_npy))
                X_train, X_val, y_train, y_val = train_test_split(sentences, labels, test_size=test_size, random_state=100)
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = np.ravel(y_train)
        self.y_val = np.ravel(y_val)
        # print(X_train.shape, y_train.shape)
        # print(len(X_train), 'train sequences')
        # print(len(X_val), 'validation sequences')


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

    def train(self):
        print('...Train...')
        start_time = time.time()
        self.model.fit(self.X_train, self.y_train)
        self.average_training_time = (time.time() - start_time)


        if os.path.exists('./tmp/models_saved/'+self.model_name+'?p='+str(self.params)+'.pkl'):
            os.remove('./tmp/models_saved/'+self.model_name+'?p='+str(self.params)+'.pkl')
        
        print('...Saving model...')
        with open('./tmp/models_saved/'+self.model_name+'?p='+str(self.params)+'.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        print('...Model Saved...') # pourquoi aussi long de saver le modèle?

    def train_tree(self,X_train,y_train):
        print('...Train...')
        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.average_training_time = (time.time() - start_time)

        if not os.path.exists('./tmp/models_saved'):
            os.makedirs('./tmp/models_saved')

        return self.model

    def predict(self):
        with open('./tmp/models_saved/classes.json', 'rb') as f:
            classes = json.load(f)
            self.target_names = [labels for key, labels in classes.items()]

        for i in range(len(self.target_names)):
            self.target_names[i] = self.target_names[i].encode('utf-8')


        self.pred = self.model.predict(self.X_val)
        self.accuracy = accuracy_score(self.y_val, self.pred)
        #self.confusion_matrix = np.array(confusion_matrix(self.y_val, self.pred), dtype=float)
        #self.classification_report = classification_report(self.y_val, self.pred, target_names=self.target_names)
        #print(self.pred)
        print('\n\n## FINISHED ##')
        print('\nresult for the ' + self.model_name + ' on validation set:')
        #print('accuracy:', self.accuracy, '\nconfusion matrix:\n', self.confusion_matrix, '\naverage_training_time:', self.average_training_time, '\nclassification report', self.classification_report)
        print('accuracy:',self.accuracy,'\naverage_training_time:',self.average_training_time)
        return [self.accuracy,self.average_training_time]


if __name__ == '__main__':

    C = 1.0
    alpha = 0.0001
    criterion = 'gini'  # 'entropy'
    n_neighbors = 5
    n_estimators = 5
    modelnames = ['reglog_l1', 'reglog_l2', 'reglog_sgd',
                  'naive_bayes',
                  'decision_tree',
                  'random_forest',
                  'bagging_reglog_l1',
                  'bagging_reglog_l2',
                  'svm_linear',
                  'knn']

    ml_models = {}
    ml_models

    X_train, X_val, y_train, y_val = prepare_data()
    modelname = modelnames[8]
    model = build_model(modelname, C)
    model, acc, average_training_time, conf = train_model(model, X_train, X_val, y_train, y_val)
    print('\n\n## FINISHED ##')
    print('\nresult for the', modelname, ':')
    print('accuracy:', acc, '\nconfusion matrix:\n', conf, '\naverage training time:', average_training_time)
