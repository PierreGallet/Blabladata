# coding: utf-8
import pandas as pd
import numpy as np
import csv
import time, operator
import os
import sys

class csv_threads():

    def __init__(self,data):
        self.data = data
        #self.df = pd.read_csv(self.data, sep=';',error_bad_lines=False)
        self.df = pd.read_csv(self.data, sep=';',error_bad_lines=False)

    def selection_ligne(self):

        print('Debut bonne selection')

        cond = self.df.apply(lambda row: (row['source_type']=='Dimelo Chat') and
        (pd.isnull(row['last_content_id'])==False), axis=1)

        self.df = self.df[cond]

        print('Fin bonne selection')

    def supprimer_cat(self,fieldnames):

        print('Debut de la suppression des colonnes inutiles')

        for column in self.df:
            if column not in fieldnames:
                print(column)
                del self.df[column]

        print('Fin de la suppression des colonnes inutiles')

    def fusion_csv(self,messages_formatted,pivot_fusion,type_fusion):

        print("Début de la fusion")


        df = pd.read_csv(messages_formatted, sep=';')

        print(self.df.shape)
        print(df.shape)
        print(self.df.columns.values)
        print(df.columns.values)

        df2 = self.df.merge(df, on=pivot_fusion, how =type_fusion)

        self.df = df2

        print("Fin de la fusion")


    def dataframe_to_csv(self,name):
        self.df.to_csv(name,sep=';',index = False)

if __name__ == '__main__':

    data_directory = './data/SFR/rawdata'
    csv_concatenated = './data/SFR/csv_concatenated.csv'
    data_file= './data/SFR/messages_formatted.csv'
    new_directory = './sfr'

    formatting = csv_sfr(csv_concatenated)
    formatting.problem_detection(3,lecture = True)
    formatting.dataframe_to_csv(data_file)
