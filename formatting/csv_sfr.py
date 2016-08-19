# coding: utf-8
import pandas as pd
import numpy as np
import csv
import time, operator
import os
import sys

def remove_bom(filename):
    fp = open(filename, 'rU')
    if fp.read(2) != '\xfe\xff':
        fp.seek(0, 0)
    return fp

def fonfon(series):
    s = series.to_string(index=False)
    s = s.encode('utf-8')
    s = s.split('\n')
    s = map(str.strip, s)
    return '\n'.join(s)

def selection_input(body):
    results=body.split('\n')
    results = results[0:3]
    results.sort(key=lambda s: -len(s))
    if not results:
        pass
    else:
        # on trouve la max_len, on prend que les messages de 0,8*max_len minimum, et on concatene.
        max_len = len(results[0])
        issue = [r for r in results if len(r)>max_len*0.7]
        if len(issue)>1:
            return ' '.join(issue)
        else:
            return issue[0]+''

class csv_sfr():

    def __init__(self,data):
        self.data = data
        #self.df = pd.read_csv(self.data, sep=';',error_bad_lines=False)
        self.df = pd.read_csv(self.data, sep=';',error_bad_lines=False)

    def bonnes_categories(self):
        #Prend un csv en entrée et rend un csv en sortie avec les bonnes catégories, et
        #ne garde que les DIMELO et conv avec body et catégorie
        print('Debut bonne categorie')
        try:
            cond = self.df.apply(lambda row: (row['source_type']=='Dimelo Chat') and (pd.isnull(row['body'])==False)
                              and (pd.isnull(row['categories'])==False) , axis=1)
        except:
            cond = self.df.apply(lambda row: (pd.isnull(row['body'])==False)
                                  and (pd.isnull(row['categories'])==False) , axis=1)

        self.df = self.df[cond]

        for index, value in self.df.categories.iteritems():
            self.df.set_value(index, 'categories',self.df.ix[index,'categories'].split(',')[0] )
            #if index%1000 ==0:
                #print (index)
        print('Fin de la correction des catégories')

    def supprimer_cat(self):


        if 'created_at' in self.df.columns.values[0]:
            self.df.columns.values[0]='created_at'
            print ('created_at est bien dedans')

        print(self.df.columns.values)
        fieldnames = ['created_at','categories','intervention_id','creator_name','body','id']
        #df_light = pd.DataFrame(index = range(0,len(self.df)),columns = fieldnames)
        i=0
        print(type(self.df.columns.values))
        print(self.df.columns)
        for column in self.df:
            if column not in fieldnames:
                print(column)
                self.df.drop(column, 1)

        print('fin de la suppression des colonnes inutiles')

    def problem_detection(self,nphrase,lecture=False):
        # Du dataframe classique à un dataframe par conversation avec catégorie, problème, conversation, intervention_id

        self.bonnes_categories()
        self.supprimer_cat()
        print(self.df.columns)
        print("Début de la détection du problème")

        # # permet d'avoir les phrases en entier, sinon elles sont coupées
        pd.options.display.max_colwidth = 3000

        self.df.creator_name=self.df.creator_name.fillna('')

        self.df.columns.values[1]='created_at'
        print(self.df.columns.values[1])
        print(self.df.columns)

        #df2 = self.df.groupby(['intervention_id','categories'],as_index=False).agg({'body':fonfon,'created_at':lambda x: x.iloc[0]})

        df2 = self.df.groupby(['intervention_id','categories'],as_index=False).agg({'body':fonfon,'id':lambda x: x.iloc[-1]})


        print("L'étape d'agrégation conversations est terminée")

        #df3 = self.df.groupby(['intervention_id','categories','creator_name'],as_index=False).agg({'body':fonfon,'created_at':lambda x: x.iloc[0]})

        df3 = self.df.groupby(['intervention_id','categories','creator_name'],as_index=False).agg({'body':fonfon})

        print("L'étape de sélection des inputs est terminée")

        df3 = df3[df3.creator_name=='']
        series = df3['body']
        series = series.apply(selection_input)
        df3 = pd.concat([df3['intervention_id'], series], axis=1)

        self.df = pd.merge(df2, df3, on='intervention_id', how='outer')

        #Je supprime la colonne created_at pour revenir au même dataframe qu'avant
        #del self.df['created_at']

        #change name of the column
        self.df = self.df.rename(columns={'categories': 'label','body_x' : 'conversation','body_y':'sentence','id':'last_content_id'})

        print("Formatting terminé")

        # fieldnames = ['label', 'sentence', 'conversation', 'intervention_id']
        #
        # print('début df_new')
        # #print len(self.df.groupby('intervention_id').size().size) prend un temps monstrueux
        # df_new = pd.DataFrame(index=range(0,self.df.groupby('intervention_id').size().size),columns = fieldnames)
        # print('fin construction df_new')
        #
        # print self.df.groupby('intervention_id').size().size
        # i=0
        # print(i)
        # for intervention in self.df.groupby('intervention_id').size().axes[0]:
        #     if i%500==0:
        #         print i
        #     # on récupère que les données correspondant à cette intervention_id
        #     subset = self.df[self.df.intervention_id == intervention]
        #     # on recupere la categorie, l'intervention_id et les 5 premiers messages client
        #     categorie = subset['categories'].values[0]
        #     intervention_id = subset['intervention_id'].values[0]
        #     results_all = '\n'.join(subset['body'].values.tolist())
        #     # on trie les messages selon leur longeur
        #     subset2 = subset[pd.isnull(self.df.creator_name)]
        #     results = subset2['body'].values.tolist()[0:nphrase]
        #     results.sort(key=lambda s: -len(s))
        #     if not results:
        #         pass
        #         i+=1
        #     else:
        #     # on trouve la max_len, on prend que les messages de 0,8*max_len minimum, et on concatene.
        #         max_len = len(results[0])
        #         issue = [r for r in results if len(r)>max_len*0.7]
        #         if len(issue)>1:
        #             issue = ' '.join(issue)
        #         else:
        #             issue = issue[0]+''
        #
        #         for colonne in fieldnames:
        #             df_new.ix[i,:] = [categorie,issue,results_all,intervention_id]
        #         i+=1
        #         if lecture == True:
        #             if i>100:
        #                 break
        # self.df= df_new

    def dataframe_to_csv(self,name):
        self.df.to_csv(name,sep=';',index = False)

if __name__ == '__main__':
    # dossier = './Scrapping_dimelo_juinjuillet'
    # #name = 'message_formated_juillet.csv'
    # test = csv_sfr('csv_concatenated.csv')
    # #test.concatenate_csv(dossier,name)
    # test.problem_detection(3,lecture = False)
    # test.dataframe_to_csv('message_formated_juillet.csv')
    #print os.listdir('./csv_directory')
    #test.check_concat_work('csv_concatenated.csv',dossier)
    #df = pd.read_csv('./data/SFR/messages_formatted.csv',sep=';',low_memory= False )
    #df.to_csv('mess_formaté_index',sep=';',index = True)

    data_directory = './data/SFR/rawdata'
    csv_concatenated = './data/SFR/csv_concatenated_test.csv'
    data_file= './data/SFR/messages_formatted_test.csv'
    new_directory = './sfr'

    formatting = csv_sfr(csv_concatenated)
    formatting.problem_detection(3,lecture = True)
    formatting.dataframe_to_csv(data_file)
