# coding: utf-8
import pandas as pd
import numpy as np
import csv
import time, operator, math

# raw_data = pd.read_csv('./data/SFR/messages.csv', sep=';')
# cond = raw_data.apply(lambda row: pd.isnull(row['creator_name']) and pd.isnull(row['body']) == False, axis=1)
# raw_data = raw_data[cond]
#
# for index, value in raw_data.intervention_id.drop_duplicates().iteritems():
#     print index, value

def formating_QA(path):
    raw_data = pd.read_csv(path, sep=';')
    # on enleve tous ce qui nous interesse pas (on g)
    cond = raw_data.apply(lambda row: pd.isnull(row['body']) == False and row['source_type'] == 'Dimelo Chat', axis=1)
    raw_data = raw_data[cond]
    progress_bar = '=>'
    i = 0

    with open('./data/SFR/messages_formated_QA.csv', 'wb') as output:
        fieldnames = ['label', 'question', 'answer', 'intervention_id']
        writer = csv.DictWriter(output,  fieldnames=fieldnames, delimiter=';')
        writer.writeheader()
        i = 0
        start_time = time.time()
        print 'there is ', len(raw_data.intervention_id.drop_duplicates().dropna()), 'conversations'
        for intervention in raw_data.intervention_id.drop_duplicates().dropna():
            # on récupère que les données correspondant à cette intervention_id
            subset = raw_data[raw_data.intervention_id == intervention]
            # on recupere la categorie, l'intervention_id et les 5 premiers messages client
            categorie = subset['categories'].values[0]
            categorie = categorie.split(',')[0]
            intervention_id = subset['intervention_id'].values[0]
            j = 0
            while j in range(len(subset['body'].values.tolist())):
                question = []
                answer = []
                while j in range(len(subset['body'].values.tolist())) and pd.isnull(subset['creator_name'].values.tolist()[j]):
                    question.append(subset['body'].values.tolist()[j])
                    j += 1
                    if len(question)>1:
                        question = [' '.join(question)]
                if j not in range(len(subset['body'].values.tolist())):
                    break
                while j in range(len(subset['body'].values.tolist())) and not pd.isnull(subset['creator_name'].values.tolist()[j]):
                    answer.append(subset['body'].values.tolist()[j])
                    j += 1
                    if len(answer)>1:
                        answer = [' '.join(answer)]
                # print '#new#', question, '\n>>>', answer
                # on ecrit tout ça dans un csv.
                if len(answer) is not 0 and len(question) is not 0:
                    writer.writerow({'label': categorie, 'question': question[0], 'answer': answer[0], 'intervention_id': intervention_id})

            if i % int(len(raw_data.intervention_id.drop_duplicates())/100) == 0:
                progress_bar = '=' + progress_bar
                print '[ ' + str(int(i*100/len(raw_data.intervention_id.drop_duplicates()))) + '% ' + progress_bar + ' ]'
            i += 1


        print 'elapsed:', time.time()-start_time
    print('...formating ended')

def formating_csv(path):
    # on recupère les données brutes
    raw_data = pd.read_csv(path, sep=';')
    # print raw_data.head()

    # on enleve tous ce qui nous interesse pas (on g)
    cond = raw_data.apply(lambda row: pd.isnull(row['creator_name']) and pd.isnull(row['body']) == False and row['source_type'] == 'Dimelo Chat', axis=1)
    raw_data = raw_data[cond]
    progress_bar = '=>'
    i = 0

    with open('./data/SFR/messages_formated.csv', 'wb') as output:
        fieldnames = ['label', 'sentence', 'conversation', 'intervention_id']
        writer = csv.DictWriter(output,  fieldnames=fieldnames, delimiter=';')
        writer.writeheader()
        i = 0
        start_time = time.time()
        for intervention in raw_data.intervention_id.drop_duplicates().dropna():
            # on récupère que les données correspondant à cette intervention_id
            subset = raw_data[raw_data.intervention_id == intervention]
            # on recupere la categorie, l'intervention_id et les 5 premiers messages client
            categorie = subset['categories'].values[0]
            categorie = categorie.split(',')[0]
            intervention_id = subset['intervention_id'].values[0]
            results = subset['body'].values.tolist()[0:3]
            results_all = '\n'.join(subset['body'].values.tolist())
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
                writer.writerow({'label': categorie, 'sentence': issue, 'conversation': results_all, 'intervention_id': intervention_id})
        print 'elapsed:', time.time()-start_time
    print('...formating ended')


def formating_bonjour(path):
    # on recupère les données brutes
    raw_data = pd.read_csv(path, sep=';')
    # print raw_data.head()

    # on enleve tous ce qui nous interesse pas (on g)
    cond = raw_data.apply(lambda row: pd.isnull(row['creator_name']) and pd.isnull(row['body']) == False and row['source_type'] == 'Dimelo Chat', axis=1)
    raw_data = raw_data[cond]
    progress_bar = '=>'
    i = 0

    with open('./data/SFR/messages_formated_bonjour.csv', 'wb') as output:
        fieldnames = ['label', 'sentence', 'conversation', 'intervention_id']
        writer = csv.DictWriter(output,  fieldnames=fieldnames, delimiter=';')
        writer.writeheader()
        i = 0
        start_time = time.time()
        for intervention in raw_data.intervention_id.drop_duplicates().dropna():
            # on récupère que les données correspondant à cette intervention_id
            subset = raw_data[raw_data.intervention_id == intervention]
            # on recupere la categorie, l'intervention_id et les 5 premiers messages client
            intervention_id = subset['intervention_id'].values[0]
            results = subset['body'].values.tolist()[0:2]
            results_all = '\n'.join(subset['body'].values.tolist())
            issue = [r for r in results if len(r)<20]
            if len(issue)>1:
                issue = ' '.join(issue)
            else:
                pass

            if i % int(len(raw_data.intervention_id.drop_duplicates())/100) == 0:
                progress_bar = '=' + progress_bar
                print '[ ' + str(int(i*100/len(raw_data.intervention_id.drop_duplicates()))) + '% ' + progress_bar + ' ]'
            i += 1
            # on ecrit tout ça dans un csv.
            if len(issue)<30 and issue:
                writer.writerow({'label': 'greetings', 'sentence': issue, 'conversation': results_all, 'intervention_id': intervention_id})
        print 'elapsed:', time.time()-start_time
    print('...formating ended')


def concat_csv(paths, output):
    """
    concatenate all csv and keep only Dimelo Chat data
    """
    l = []
    for path, sep in paths.items():
        raw_data = pd.read_csv(path, sep=sep)
        cond = raw_data.apply(lambda row: pd.isnull(row['body']) == False and row['source_type'] == 'Dimelo Chat', axis=1)
        raw_data = raw_data[cond]
        del raw_data['title']
        l.append(raw_data)
    pd.concat(l).to_csv(output, sep=';', index=False)


def select_categories(path):

    raw_data = pd.read_csv(path, sep=';')
    cat = {}
    for label in raw_data.label.drop_duplicates():
        subset = raw_data[raw_data.label == label]
        cat[label] = len(subset)
    with open('./data/SFR/messages_formated_cat.csv', 'wb') as output:
        fieldnames = ['label', 'sentence']
        writer = csv.DictWriter(output,  fieldnames=fieldnames, delimiter=';')
        writer.writeheader()
        for label, nb in sorted(cat.items(), key=operator.itemgetter(1), reverse=True)[0:11]:
            if label == 'Service client':
                pass
            else:
                print label, nb
                subset = raw_data[raw_data.label == label]['sentence'].values
                for issue in subset[:500]:
                    writer.writerow({'label': label, 'sentence': issue})
            # elif label in ["Changer d'offre"]:
            #     print label, nb
            #     subset = raw_data[raw_data.label == label]['sentence'].values
            #     for issue in subset:
            #         writer.writerow({'label': label, 'sentence': issue})



if __name__ == '__main__':

    paths = {}
    paths['./data/SFR/autres/messages_juillet.csv'] = ','
    paths['./data/SFR/autres/messages_22june.csv'] = ','
    paths['./data/SFR/autres/messages_janv_mars.csv'] = ','
    # paths['./data/SFR/autres/messages_1june.csv'] = ';'
    # concat_csv(paths, './data/SFR/messages_all.csv')

    # formating_csv('./data/SFR/messages_all.csv')
    # formating_bonjour('./data/SFR/messages_all.csv')
    select_categories('./data/SFR/messages_formated.csv')
    # formating_QA('./data/SFR/autres/messages_1june.csv')
