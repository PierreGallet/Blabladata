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

    def changer_nom_colonne(self,dict_nom):

        print('Debut du changement de nom')

        self.df = self.df.rename(columns=dict_nom)

        print('Fin du changement de nom')


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

    def csv_size(self):
        print(self.df.shape[0])

    def csv_categories(self,column,name):

        df2 = self.df.groupby(column).size()
        print (df2)
        if not os.path.isdir('./data/stats'):
            os.mkdir('./data/stats')
        df2.to_csv(name,sep=';')

    def csv_visualisation(self,column):

        if not os.path.exists('./data/scripts_categories'):
            os.makedirs('./data/scripts_categories')

        df2 = self.df.groupby(column).size()
        df2 = df2.sort_values(ascending = False)

        print(type(df2[0:19]))

        itera = df2

        for index, value in itera.iteritems():

            with open('./data/scripts_categories/'+index+'_'+str(value)+'.txt','wb') as fichier:

                fichier.write(index+":"+str(value)+"\n\n")

                df = pd.read_csv('./data/SFR/messages_formated_cat.csv', sep=';',error_bad_lines=False)
                group = df.groupby('custom_motif')
                for index2,value2 in group['conversation'].get_group(index).iteritems():
                    fichier.write(value2)
                    fichier.write('\n\n\n ------ Nouvelle discussion --------- \n\n\n')

    def csv_top_cat(self,top,column):
        df2 = self.df.groupby([column]).size()
        df2 = df2.order(ascending=False)

        toplabel = list(df2.index.values)
        toplabel = toplabel [0:top]

        cond = self.df.apply(lambda row: (row[column] in toplabel),axis=1)

        self.df = self.df[cond]

    def sep_motifs(self,column):
        print(self.df[column])
        labels = self.df[column]
        labels = labels.apply(lambda labe: labe.split('-'))
        for index,value_serie in labels.iteritems():
            for key,value in enumerate(value_serie):
                self.df.ix[index,column+'_'+str(key+1)] = value




if __name__ == '__main__':

    data_directory = './data/SFR/rawdata'
    csv_concatenated = './data/SFR/csv_concatenated.csv'
    data_file= './data/SFR/messages_formatted.csv'
    new_directory = './sfr'

    # formatting = csv_sfr(csv_concatenated)
    # formatting.problem_detection(3,lecture = True)
    # formatting.dataframe_to_csv(data_file)

    test = csv_threads('./data/SFR/messages_motifs.csv')
    #test.csv_categories('label','./data/stats/effectif_motifs.csv')
    test.csv_visualisation('label')
