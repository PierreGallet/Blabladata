# coding: utf-8
import pandas as pd
import numpy as np
import csv
import time, operator
import os
import sys

class create_tree():

    def __init__(self):
        self.tree ={};
        self.nodes =[];

    def add_node_1(self,intent,message):
        self.nodes.append([intent,message])

    def add_node_2(self,nodes):
        for element in nodes:
            self.nodes.append(element)

    def initialize_tree(self,num_nodes):
        for element in num_nodes:
            node = list(self.nodes[element])
            node.append({})
            self.tree[str(element)]= node

    def add_branches(self,num_nodes,place):
        nodes=self.tree
        for element in place:
            nodes=nodes[str(element)][2]
        for element in num_nodes:
            node = list(self.nodes[element])
            node.append({})
            nodes[str(element)]= node

if __name__ == '__main__':

    test=create_tree()
    test.add_node_2([["intent a",'mess a'],["intent b", "mess b"],["intent c", "mess c"],["intent d", "mess d"]])
    test.initialize_tree([0,1])
    test.add_branches([2,3,1],[0])
    test.add_branches([0,2],[0,2])
    print(test.nodes)
    print(test.tree)
