# coding: utf8
from __future__ import division
import pickle, operator, os
import pandas as pd
import numpy as np
import sys
from formatting.csv_threads import csv_threads
from preprocessing.parse import preprocessing
from classifiers import deep_learning, machine_learning
from embeddings import tfidf, word2vec
from sklearn.metrics.pairwise import cosine_similarity
from ner.train import ner
from classifiers import train
from copy import copy
from sklearn.cross_validation import train_test_split
data_directory = './data/SFR/messages_formated_cat.csv'

def merge_wrong_branch(tree):
    try: #probleme du bug impossible à régler
        for element in tree.keys():
            if tree[element]!={}:
                dico=tree[element]
                count = len(dico.keys())
                if count == 1:
                    del tree[element]
                    for ele in dico.keys():
                        new_element = element+'-'+ele
                        tree[new_element]=dico[ele]
                    merge_wrong_branch(tree)
                else:
                    merge_wrong_branch(tree[element])
            else:
                pass
    except:
        pass
    return tree



def tree_function(liste,dico={}):
    if len(liste)>1:
        dico={}
        dico[liste[0]]=tree_function(liste[1:],dico)
        #print(1,dico)
        return dico
    else:
        dico={}
        dico[liste[0]]={}
        #print(2,dico)
        return dico

def ajout(liste,dico):
    try:
        if liste[0] in dico.keys():
            dico[liste[0]] = ajout(liste[1:],dico[liste[0]])
            return dico
        else:
            dico2 = tree_function(liste,dico={})
            dico.update(dico2)
            return dico
    except:
        print('Erreur, la branche existe déjà')

def concatenate_branches(liste,dico):
    for element in liste:
        dico = ajout(element,dico)
    return dico

def merge_parents_childs(parents,childs):
    merge =[]
    for branch in childs:
        merge.append(parents+branch)
    for index,element in enumerate(merge):
        string = '-'.join(element)
        merge[index]=string
    return merge

def liste_position(tree,position,liste):
    for element in tree.keys():
        if tree[element]!={}:
            position2 = copy(position)
            position.append(element)
            liste_position(tree[element],position,liste)
            position = copy(position2)
        else:
            position2 = copy(position)
            position.append(element)
            liste.append(position)
            position = copy(position2)
    return liste

def liste_position_main(main_tree,parents,position_global,boolean):

    if boolean == True:
        position_global.append(merge_parents_childs([],liste_position(main_tree,[],[])))
        boolean = False

    for element in main_tree.keys():
        if main_tree[element]!={}:
            parents2 = copy(parents)
            parents.append(element)
            position_global.append(merge_parents_childs(parents,liste_position(main_tree[element],[],[])))
            liste_position_main(main_tree[element],parents,position_global,boolean)
            parents=copy(parents2)
        else:
            #position_global.append(merge_parents_childs(parents,[[element]]))
            pass
    return position_global

def deno_commun(liste):
    if len(liste)==1:
        return liste[0]
    else:
        for ele in liste[1:]:
            if ele[0] != liste[0][0]:
                return''
        for ele in liste[1:]:
            if ele[2] != liste[0][2]:
                return liste[0][0]
        for ele in liste[1:]:
            if ele[4] != liste[0][4]:
                return liste[0][0]+'-'+liste[0][2]

def dico_node(liste_node):
    dico={}
    for ind,ele in enumerate(liste_node):
        key = deno_commun(ele)
        dico[key]=ele
    return dico

def somme_str(raw):
    if np.isnan(raw[2]):
        return int(str(int(raw[0]))+str(int(raw[1])))
    else:
        return int(str(int(raw[0]))+str(int(raw[1]))+str(int(raw[2])))

def concat_labs(labels):
    new_labels=np.zeros((labels.shape[0], labels.shape[1]+1))
    new_labels[:,0:new_labels.shape[1]-1]=labels
    col = np.apply_along_axis( somme_str, axis=1, arr=labels )
    new_labels[:,new_labels.shape[1]-1]=col
    return col




class tree_classifier():

    def __init__(self, data_directory, word_label=True):
        #initialize
        self.data_directory = data_directory
        self.df = pd.read_csv(self.data_directory, sep=';')
        self.path_sentences = './data/inputs/tfidf/sentences.npy'
        self.path_labels = './data/inputs/tfidf/labels.npy'
        self.path_labels_1 = './data/inputs/tfidf/labels_1.npy'
        self.path_labels_2 = './data/inputs/tfidf/labels_2.npy'
        self.path_labels_3 = './data/inputs/tfidf/labels_3.npy'
        self.label_list=['label_1','label_2','label_3']

    def vectorize_and_split(self,test_size = 0.20):
        # compute tfidf and transform the sentences into tfidf vectors, with feature selection (best tfidf weights)
        self.tfidf = tfidf.tfidf()
        #if not os.path.exists('./tmp/tfidf.pkl'):
        self.tfidf.train('./data/inputs/sentences.txt')
        self.tfidf.format_input_tree()

        with open(self.path_sentences, 'rb') as sentences_npy:
            with open(self.path_labels_1, 'rb') as labels_1_npy:
                with open(self.path_labels_2, 'rb') as labels_2_npy:
                    with open(self.path_labels_3, 'rb') as labels_3_npy:
                            sentences = np.array(np.load(sentences_npy))
                            labels_1 = np.array(np.load(labels_1_npy))
                            labels_2 = np.array(np.load(labels_2_npy))
                            labels_3 = np.array(np.load(labels_3_npy))
                            self.labels = np.concatenate((labels_1,labels_2,labels_3),axis=1)
                            X_train, X_val, y_train, y_val = train_test_split(sentences, self.labels, test_size=test_size)

                            self.X_train = X_train
                            self.X_val = X_val
                            self.y_train = y_train
                            self.y_val = y_val

        print(X_train.shape, y_train.shape)
        print(len(X_train), 'train sequences')
        print(len(X_val), 'validation sequences')

    def generate_tree(self):
        # generate the tree representing the motifs
        label = self.df['label']
        label = label.replace({'F-1-4':'5-1-4'},regex=True)
        label = list(label.unique())
        label = [labe.split('-') for labe in label]
        self.tree = concatenate_branches(label,{})
        print(self.tree)

    def generate_label_at_node(self):
        self.tree=merge_wrong_branch(self.tree)
        self.node = liste_position_main(self.tree,[],[],True)
        self.diconode = dico_node(self.node)
        print(self.diconode)
        print(len(self.node))

    def train_node(self,node):

        print('Debut entrainement du noeud',node)
        label_selection = self.diconode[node]
        for index,ele in enumerate(label_selection):
            ele = ele.replace('-','')
            label_selection[index]=int(ele)
        see = concat_labs(self.y_train)
        cond = np.in1d(see,label_selection)
        y_train_select = self.y_train[cond]
        X_train_select = self.X_train[cond]
        print("Taille de la selection d'entrainement du noeud :",X_train_select.shape)

        node = node.split('-')
        if node ==['']:
            num_label=0
        else:
            num_label = len(node)
        y_train_select=y_train_select[:,num_label]

        ml = machine_learning.machine_learning()
        for model_name, params_list in ml_models.items():
            #results[model_name]={}
            for params in params_list:
                ml.build(model_name, params)
                result = ml.train_tree(X_train_select,y_train_select)
                #results[model_name][params] = ml.predict()
        print("Fin d'entrainement du noeud",node)
        return num_label,cond,result


    def train_tree_ml(self):

        self.models = {}
        for key,value in self.diconode.iteritems():
            self.models[key] = self.train_node(key)

    def classifier_tree_ml(self):

        self.dico_prediction = {}
        for key,value in self.models.iteritems():
            self.dico_prediction[key] = value[2].predict(self.X_val)

    def predict_node(self,pred_actuel,i):
        if pred_actuel in self.diconode.keys():
            pred_suiv = str(int(self.dico_prediction[pred_actuel][i]))
            pred_actuel = pred_actuel+'-'+pred_suiv
            if pred_actuel[0]=='-':
                pred_actuel = pred_actuel[1:]
            return self.predict_node(pred_actuel,i)
        else:
            return pred_actuel.split('-')

    def tree_prediction(self):
        self.prediction = np.zeros((self.y_val.shape[0],3))
        for i in range(0,self.y_val.shape[0]):
            liste = self.predict_node('',i)
            for index,ele in enumerate(liste):
                self.prediction[i,index]=ele
            for j in range(len(liste),3):
                self.prediction[i,j]=np.nan

    def accuracy_tree(self):
        conformite = []
        for i in range(0,self.prediction.shape[0]):
            conformite.append((self.y_val[i,0]==self.prediction[i,0] and self.y_val[i,1]==self.prediction[i,1] and self.y_val[i,2]==self.prediction[i,2]))
        print("Précision:",sum(conformite)/self.prediction.shape[0])

    def accuracy_node(self,node):
        y_prediction = self.dico_prediction[node]
        print(y_prediction.shape)

        label_selection = self.diconode[node]
        print(label_selection)
        #for index,ele in enumerate(label_selection):
            #ele = ele.replace('-','')
            #label_selection[index]=int(ele)
        see = concat_labs(self.y_val)
        cond = np.in1d(see,label_selection)

        print(self.y_val.shape)
        print(cond.shape)

        y_val_select = self.y_val[cond]

        node = node.split('-')
        if node ==['']:
            num_label=0
        else:
            num_label = len(node)
        y_val_select=y_val_select[:,num_label]

        y_prediction_select = y_prediction[cond]
        conformite = (y_val_select==y_prediction_select)
        accuracy_node = np.sum(conformite)/y_val_select.shape[0]
        print(node,accuracy_node)


if __name__ == '__main__':

    deep = False
    # dictionary that links machine learning models to their parameters
    ml_models = {}
    basic = [10**x for x in range(-3,3)]
    #ml_models['reglog_l1'] = [10]  # C
    ml_models['reglog_l2'] = [1]  # C
    #ml_models['reglog_sgd'] = 0.0001  # alpha
    #ml_models['naive_bayes'] = ''
    #ml_models['decision_tree'] = 'gini'  # entropy
    #ml_models['random_forest'] = [100]  # nb_estimator
    #ml_models['bagging_reglog_l1'] = 5  # nb_estimator
    #ml_models['bagging_reglog_l2'] = 5  # nb_estimator
    #ml_models['svm_linear'] = 1.0  # C
    #ml_models['knn'] = 5  # nb_neighbors

    data_directory = './data/SFR/messages_formated_cat.csv'
    data_directory_2 = './data/tree_classifier/messages_motifs.csv'

    #Selection des data
    #On s'arrange pour ne garder que les deux colonnes sentence et label du csv
    print (sys.version)
    data = csv_threads(data_directory)
    data.supprimer_cat(['custom_code_motif','sentence'])
    data.changer_nom_colonne({'custom_code_motif':'label'})
    print (data.df.columns.values)
    cond = data.df.apply(lambda row:(pd.isnull(row['label'])==False), axis=1)
    data.df = data.df[cond]
    data.csv_top_cat(14,'label')
    print(data.df)
    data.sep_motifs('label')
    print(data.df)
    # data.df = data.df.replace({'F': '5'}) # on remplace le label 'F par 5 pour le numpyarray'
    # data.dataframe_to_csv(data_directory_2)

    # #Preprocessing
    # preprocessing = preprocessing(data_directory_2)
    # preprocessing.csv_multi_motifs(word_label=False)
    #
    # # # Classification
    #
    # classifier = tree_classifier(data_directory_2)
    # classifier.vectorize_and_split()
    # classifier.generate_tree()
    # classifier.generate_label_at_node()
    # classifier.train_tree_ml()
    # classifier.classifier_tree_ml()
    # classifier.tree_prediction()
    # classifier.accuracy_tree()
    # classifier.accuracy_node('')
    # classifier.accuracy_node('1')
    # classifier.accuracy_node('2')
    # classifier.accuracy_node('4')
    # classifier.accuracy_node('2-2')
    # classifier.accuracy_node('4-4')

    # and put data in sentence.txt and label.txt
    #classifier.train_ml()  # does the training
