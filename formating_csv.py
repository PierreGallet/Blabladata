# coding: utf-8
import pandas as pd
import numpy as np
import csv
import time

def formating_csv(path):
    # on recupère les données brutes
    raw_data = pd.read_csv(path, sep=';')


    # on enleve tous ce qui nous interesse pas (on g)
    cond = raw_data.apply(lambda row: pd.isnull(row['creator_name']) and pd.isnull(row['body']) == False, axis=1)
    raw_data = raw_data[cond]
    progress_bar = '=>'
    i = 0

    with open('./data/SFR/messages_formated.csv', 'wb') as output:
        fieldnames = ['label', 'sentence']
        writer = csv.DictWriter(output,  fieldnames=fieldnames, delimiter=';')
        writer.writeheader()

        for intervention in raw_data.intervention_id.drop_duplicates()[1:]:
            # on récupère que les données correspondant à cette intervention_id
            subset = raw_data[raw_data.intervention_id == intervention]
            # on recupere la categorie, l'intervention_id et les 5 premiers messages client
            categorie = subset['categories'].values[0]
            #### intervention_id = subset['intervention_id'].values[0]
            results = subset['body'].values.tolist()[0:3]
            #### results_all = '\n'.join(subset['body'].values.tolist())
            # on trie les messages selon leur longeur
            results.sort(key=lambda s: -len(s))
            # on trouve la max_len, on prend que les messages de 0,8*max_len minimum, et on concatene.
            max_len = len(results[0])
            issue = [r for r in results if len(r)>max_len*0.7]
            if len(issue)>1:
                issue = ' '.join(issue)
            else:
                issue = issue[0]

            if i % int(len(raw_data.intervention_id.drop_duplicates())/100) == 0:
                progress_bar = '=' + progress_bar
                print '[ ' + str(int(i*100/len(raw_data.intervention_id.drop_duplicates()))) + '% ' + progress_bar + ' ]'
            i += 1
            # on ecrit tout ça dans un csv.
            if len(issue)>15:
                writer.writerow({'label': categorie, 'sentence': issue})
    print('...formating ended')
