
# coding: utf-8

# In[5]:

import pickle
import re
import string
import collections
#import pattern
import numpy as np
from nltk.stem.snowball import FrenchStemmer
from nltk.corpus import stopwords
from nltk.tokenize.regexp import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfTransformer


# In[6]:

def mailslabelises_to_dictionnaire(mail_labelise):
    # Input : Liste de 2-uple contenant (mail,label)
    # 0utput : Dictionnaire = liste de 2-uple avec l'ensemble des mots de tous les mails (mot,count)

    mail = [mail_labelise[i][0] for i in range(0,len(mail_labelise))]
    label = [mail_labelise[i][1] for i in range(0,len(mail_labelise))]

    not_letters_or_digits = u'!"#%\'()*+,-./:;<=>?@[\]^_`{|}~'
    translate_table = dict((ord(char),u' ') for char in not_letters_or_digits)
    count = allcollections.Counter()
    stemmer = FrenchStemmer()

    # Processing de tous les mails en même temps pour avoir un dictionnaire unique
    for i in range(0,len(mail)):
        a=mail[i].translate(translate_table) # On enleve la ponctuation
        a=a.lower() # On met tout en minuscule
        a=a.split() # On répartit selon les espaces
        a=[stemmer.stem(word) for word in a]
        a=collections.Counter(a) # On met sous la forme de Counter
        count=count+a # On ajoute

    dico = count.most_common()

    # Enlever les mots qui apparaissent moins de 10 fois
    for i in range(0,len(dico)):
        if dico[i][1]==10:
            del dico[i:len(dico)]
            break

    # Enlevez les chiffres
    i=0
    while i<len(dico):
        if (dico[i][0]).isdigit():
            del dico[i]
            i-=1
        i+=1

    # Effacer automatiquement les stop words
    stop = stopwords.words('french')
    i=0
    while i<len(dico):
        if dico[i][0] in stop:
            del dico[i]
            i-=1
        i+=1

    # Effacer manuellement les stop words
    i=0
    while i<len(dico):
        print dico[i][0],':',dico[i][1]
        eff= raw_input('Effacer ce mot y/n :')
        if eff=='y':
            del dico[i]
            i-=1
        if eff=='q':
            break
        i+=1

    return dico


# In[7]:

def mail2vec_freq(mail,dico_global):
    # Input: 1 mail +  dictionnaire = liste de 2-uple (mot, count)
    # Output: retourne array colonne contenant la fréquence d'apparition des mots du dico_global dans le mail

    not_letters_or_digits = u'!"#%\'()*+,-./:;<=>?@[\]^_`{|}~'
    translate_table = dict((ord(char),u' ') for char in not_letters_or_digits)
    count=collections.Counter()
    stemmer = FrenchStemmer()
    mail=mail.translate(translate_table) # On enleve la ponctuation
    mail=mail.lower() # On met tout en minuscule
    mail=mail.split() # On répartit selon les espaces
    mail=[stemmer.stem(word) for word in mail]
    mail=collections.Counter(mail)
    mail=mail.items()

    mots_dic = [dico_global[i][0] for i in range(0,len(dico_global))]
    vec_mail=np.zeros((len(dico_global),1))

    for i in range(0,len(mail)):
        if mail[i][0] in mots_dic:
            j=mots_dic.index(mail[i][0])
            vec_mail[j]=mail[i][1]
    return vec_mail


# In[8]:

def data_matrix(mail_label,dico_global):
    # Input : Liste de 2-uple contenant (mail,label)
    # Output : array(matrix) n x (p+1), avec n nombre de mails, et p nombre de mots dans le dictionnaire, et p+1 ème
    # feature correspond au label
    #dico_global=mailslabelises_to_dictionnaire(mail_label)
    matrix=np.zeros((len(mail_label),len(dico_global)))
    label=np.zeros((len(mail_label),1))
    for i in range(0,len(mail_label)):
        raw_i=np.zeros((1,len(dico_global)))
        freq=mail2vec_freq(mail_label[i][0],dico_global)
        raw_i[0,0:len(dico_global)]=freq.transpose()
        label[i] = mail_label[i][1]
        matrix[i,]=raw_i
    return matrix,label


# In[9]:

def data_matrix_tfidf(mail_label,dico_global):
    [mat1,Y] = data_matrix(mail_label,dico_global)
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(mat1)
    tfidf=tfidf.toarray()
    return tfidf,Y


# In[ ]:
