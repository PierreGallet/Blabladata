# coding: utf-8

import pandas as pd
import numpy as np
import csv
import time, operator
import os, math
import concatenate_csv

class csv_stat():

    def __init__(self,data):
        self.data = data
        self.df = pd.read_csv(self.data, sep=';',low_memory=False)

    def visualisation(self):
        """ Permet de bien visualiser les catégories en créant des fichiers textes correspondant"""

        if not os.path.exists('Scripts_categories'):
            os.makedirs('Scripts_categories')

        # Fonction qui affiche au fur et à mesure les éléments d'une certaine catégorie
        # try:
        #     cond = self.df.apply(lambda row: (row['source_type']=='Dimelo Chat') and (pd.isnull(row['body'])==False) and (pd.isnull(row['categories'])==False) , axis=1)
        # except:
        #     cond = self.df.apply(lambda row: (pd.isnull(row['body'])==False) and (pd.isnull(row['categories'])==False) , axis=1)
        #
        # raw_data = self.df[cond]

        raw_data = self.df

        count = dict()

        series= raw_data.intervention_id.drop_duplicates()

        for index, value in series.iteritems():
            if raw_data.categories[index] not in count.keys():
                count[raw_data.categories[index]]=1
            else:
                count[raw_data.categories[index]]+=1

        func = operator.itemgetter(1)
        result = sorted(count.items(), key= lambda s: -func(s))

        print result

        for nbre_categorie in range(0,30) :

            #nbre_categorie = raw_input('\nTaper le numéro de la catégorie')

            categorie = result[nbre_categorie][0]
            effectif = result[nbre_categorie][1]

            with open('Scripts_categories/'+str(nbre_categorie)+'.txt','wb') as fichier:

                fichier.write(categorie+":"+str(effectif)+'\n\n')

                    # On a la catégorie, on parcourt chaque conv de cette cat

                cond_bis = raw_data.apply(lambda row: (row['categories']==categorie), axis=1)
                raw_cat = raw_data[cond_bis]

                compteur=1

                for intervention in raw_cat.intervention_id.drop_duplicates():
                    fichier.write('Categorie : '+categorie+' Discussion :'+ str(compteur) + '\n\n')
                    subset = raw_data[raw_data.intervention_id == intervention]
                    for key, value in subset['body'].iteritems():
                        if str(type(raw_data.creator_name[key])) == "<type 'float'>":
                            fichier.write("Client dit : "+ value+'\n')
                        else:
                            fichier.write(raw_data.creator_name[key]+"dit :"+value+'\n')
                    fichier.write('\n\n\n ------ Nouvelle discussion --------- \n\n\n')
                    compteur += 1

                fichier.write('\n\n\n --------- NEW CATEGORIE ----------\n\n\n')

    def list_of_categories(self):

        '''Take csv and return a csv with list of categories and effectif'''

        #The list of all categories
        # try:
        #     cond = self.df.apply(lambda row: (row['source_type']=='Dimelo Chat') and (pd.isnull(row['body'])==False) and (pd.isnull(row['categories'])==False) , axis=1)
        # except:
        #     cond = self.df.apply(lambda row: (pd.isnull(row['body'])==False) and (pd.isnull(row['categories'])==False) , axis=1)
        #
        # raw_data = raw_data[cond]

        raw_data = self.df

        count = dict()

        series= raw_data.intervention_id.drop_duplicates()

        for index, value in series.iteritems():
            if index%1000==0:
                print index
            # try:
            #     series[index] = series[index].split(',')[0]
            # except AttributeError:
            #     pass
            # raw_data.categories[index] = raw_data.categories[index].split(',')[0]

            if raw_data.label[index] not in count:
                count[raw_data.label[index]]=1
            else:
                count[raw_data.label[index]]+=1

        func = operator.itemgetter(1)
        result = sorted(count.items(), key= lambda s: -func(s))

        with open('liste_categories.csv', 'wb') as output:
            fieldnames = ['label','effectif']
            writer = csv.DictWriter(output,  fieldnames=fieldnames, delimiter=';')
            writer.writeheader()

            for cat, eff in result:
                writer.writerow({'label':cat,'effectif':eff})

    def nbre_conv(self):
        """ Nombre de conversation en données """
        self.df.intervention_id.drop_duplicates().dropna().size

if __name__ == '__main__':
    test = csv_stat('message_formated_juillet.csv')
    print test.nbre_conv()
    test.list_of_categories()
