# coding: utf-8
import pandas as pd
import numpy as np
import csv
import time, operator
import os
import sys
import shutil
from create_tree import create_tree

def somme_list_str(liste):
    res =''
    for ele in liste:
        res+=ele
    return res

def list_node(tree,list_n,context,first_node):

    if first_node == True:
        list_n.append([context,tree.keys()])
        first_node = False

    for key in tree.keys():
        if tree[key][2] != {}:
            context_copy = context
            context = context +'_'+ key
            list_n.append([context,tree[key][2].keys()])
            list_node(tree[key][2],list_n,context,first_node)
            context=context_copy
        else:
            pass
    return list_n

class create_data():

    def __init__(self,tree):

        self.new_directory = './treevertical/data'
        if not os.path.exists(self.new_directory):
            os.mkdir(self.new_directory)

        self.data = './treevertical/data/rawdata.csv'
        if not os.path.isfile(self.data):
            self.df = pd.DataFrame({},columns = ['phrase','context','intent'])
        else:
            self.df = pd.read_csv(self.data, sep=';')
        print(self.df)

        self.tree = tree


    def add_sample(self):
        phrase= raw_input("Rentrer la phrase: ")
        context = raw_input("Rentrer le contexte: ")
        intent= raw_input("Rentrer intent: ")
        self.df=self.df.append({'phrase':phrase,'context':context,'intent':intent},ignore_index=True)

    def dataframe_to_csv(self):
        self.df.to_csv(self.data,sep=';',index = False)

    def add_samples_node(self,context,label):
        phrase= raw_input("Rentrer la phrase pour context "+self.display_context_intent(context,label)[0]+" et intent "+self.display_context_intent(context,label)[1]+ " :")

        while phrase!='n':
            if phrase == 'di':
                print self.df.tail(5)
            elif phrase =='de':
                self.df = self.df.drop(self.df.index[len(self.df)-1])
                print self.df.tail(4)
            elif phrase == 'q':
                self.dataframe_to_csv()
                print self.df
                break
            else:
                self.df=self.df.append({'phrase':phrase,'context':self.display_context_intent(context,label)[0],'intent':self.display_context_intent(context,label)[1]},ignore_index=True)
            phrase= raw_input("Rentrer la phrase pour context "+self.display_context_intent(context,label)[0]+" et intent "+self.display_context_intent(context,label)[1]+ " :")


    def parcourir_arbre(self):
        liste_nodes = list_node(self.tree,[],'0',True)
        print(liste_nodes)
        print('liste noeuds',liste_nodes)
        for ele in liste_nodes:
            context=ele[0]
            for label in ele[1]:
                #print(self.display_context_intent(context))
                self.add_samples_node(context,label)
                print self.df.tail(5)

    def display_context_intent(self,context,intent):
        context=context[2:]
        longue = len(context)
        i=0
        nodes=self.tree
        while i<longue-2:
            element = context[i]
            nodes=nodes[element][2]
            i=i+2
        if len(context)==0:
            return ['Context départ',nodes[intent][0]]
        else:
            return [nodes[context[longue-1]][0],nodes[context[longue-1]][2][intent][0]]




if __name__ == '__main__':

    test=create_tree()
    test.add_node_2([["Activer ma carte SIM",'mess a'],["Commander ma carte SIM (pas encore arrivé)", "mess b"],["Débloquer votre carte SIM (code PUK erroné)", "mess c"],["Autre problème Carte SIM", "mess d"]])
    #test.add_node_2([["intent a",'mess a'],["intent b", "mess b"],["intent c", "mess c"],["intent d", "mess d"]])
    test.initialize_tree([1,2,3,4])
    # test.add_branches([3,4,2],[1])
    # test.add_branches([1,3],[1,3])
    tree=test.tree

    test = create_data(tree)
    test.parcourir_arbre()
    test.dataframe_to_csv()

    # test=create_data(tree)
    # print(test.df)
    # print(type(test.df.iloc[7,1]))
