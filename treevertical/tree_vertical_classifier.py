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
from create_tree import create_tree

class tree_vertical_classifier():

    def __init__(self,tree,data):
        self.tree = tree
        self.df = pd.read_csv(data, sep=';')

    def sep_motifs(self,column):
        print(self.df[column])
        labels = self.df[column]
        labels = labels.apply(lambda labe: labe.split('-'))
        for index,value_serie in labels.iteritems():
            for key,value in enumerate(value_serie):
                self.df.ix[index,column+'_'+str(key+1)] = value
                


if __name__ == '__main__':
        test=create_tree()
        test.add_node_2([["intent a",'mess a'],["intent b", "mess b"],["intent c", "mess c"],["intent d", "mess d"]])
        test.initialize_tree([0,1])
        test.add_branches([2,3,1],[0])
        test.add_branches([0,2],[0,2])
        print(test.tree)
