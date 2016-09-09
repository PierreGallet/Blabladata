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
        list_n.append(['',tree.keys()])
        first_node = False

    for key in tree.keys():
        if tree[key][2] != {}:
            context_copy = context
            context = context + key
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
        phrase= raw_input("Rentrer la phrase pour context "+context+" et intent "+label+ " :")
        while phrase!='next':
            self.df=self.df.append({'phrase':phrase,'context':context,'intent':label},ignore_index=True)
            phrase= raw_input("Rentrer la phrase pour context "+context+" et intent "+label+ " :")

    def parcourir_arbre(self):
        liste_nodes = list_node(self.tree,[],'',True)
        for ele in liste_nodes:
            context=ele[0]
            for label in ele[1]:
                self.add_samples_node(context,label)




if __name__ == '__main__':

    test = create_tree()
    test.add_node_2([["intent a",'mess a'],["intent b", "mess b"],["intent c", "mess c"],["intent d", "mess d"]])
    test.initialize_tree([0,1])
    test.add_branches([2,3,1],[0])
    test.add_branches([0,2],[0,2])
    tree = test.tree
    print(tree)

    test = create_data(tree)
    test.parcourir_arbre()
    test.dataframe_to_csv()
