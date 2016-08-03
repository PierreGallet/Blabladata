# coding: utf-8
import pandas as pd
import numpy as np
import csv
import time, operator
import os
import concatenate_csv

class csv_sfr():

    def __init__(self,data):
        self.data = data
        self.df = pd.read_csv(self.data, sep=';',low_memory=False)

    def bonnes_categories(self):
        #Prend un csv en entrée et rend un csv en sortie avec les bonnes catégories, et
        #ne garde que les DIMELO et conv avec body et catégorie

        try:
            cond = self.df.apply(lambda row: (row['source_type']=='Dimelo Chat') and (pd.isnull(row['body'])==False)
                              and (pd.isnull(row['categories'])==False) , axis=1)
        except:
            cond = self.df.apply(lambda row: (pd.isnull(row['body'])==False)
                                  and (pd.isnull(row['categories'])==False) , axis=1)

        self.df = self.df[cond]

        for index, value in self.df.categories.iteritems():
            self.df.set_value(index, 'categories',self.df.ix[index,'categories'].split(',')[0] )
            if index%1000 ==0:
                print index

    def supprimer_cat(self):

        fieldnames = ['created_at','categories','intervention_id','creator_name','body']
        #df_light = pd.DataFrame(index = range(0,len(self.df)),columns = fieldnames)
        for column in self.df:
            if column not in fieldnames:
                del self.df[column]

    def problem_detection(self,nphrase,lecture=False):
        # Du dataframe classique à un dataframe par conversation avec catégorie, problème, conversation, intervention_id

        self.bonnes_categories()
        self.supprimer_cat()

        fieldnames = ['label', 'sentence', 'conversation', 'intervention_id']
        df_new = pd.DataFrame(index=range(0,len(self.df.intervention_id.drop_duplicates())),columns = fieldnames)

        print self.df.intervention_id.drop_duplicates().dropna().size
        i=0
        for intervention in self.df.intervention_id.drop_duplicates().dropna():
            print i
            # on récupère que les données correspondant à cette intervention_id
            subset = self.df[self.df.intervention_id == intervention]
            # on recupere la categorie, l'intervention_id et les 5 premiers messages client
            categorie = subset['categories'].values[0]
            intervention_id = subset['intervention_id'].values[0]
            results_all = '\n'.join(subset['body'].values.tolist())
            # on trie les messages selon leur longeur
            subset2 = subset[pd.isnull(self.df.creator_name)]
            results = subset2['body'].values.tolist()[0:nphrase]
            results.sort(key=lambda s: -len(s))
            if not results:
                pass
                i+=1
            else:
            # on trouve la max_len, on prend que les messages de 0,8*max_len minimum, et on concatene.
                max_len = len(results[0])
                issue = [r for r in results if len(r)>max_len*0.7]
                if len(issue)>1:
                    issue = ' '.join(issue)
                else:
                    issue = issue[0]+''

                for colonne in fieldnames:
                    df_new.ix[i,:] = [categorie,issue,results_all,intervention_id]
                i+=1
                if lecture == True:
                    if i>100:
                        break
        self.df= df_new

    def dataframe_to_csv(self,name):
        self.df.to_csv(name,sep=';',index = False)

if __name__ == '__main__':
    dossier = './Scrapping_dimelo_juinjuillet'
    #name = 'message_formated_juillet.csv'
    test = csv_sfr('csv_concatenated.csv')
    #test.concatenate_csv(dossier,name)
    test.problem_detection(3,lecture = False)
    test.dataframe_to_csv('message_formated_juillet.csv')
    #print os.listdir('./csv_directory')
    #test.check_concat_work('csv_concatenated.csv',dossier)
