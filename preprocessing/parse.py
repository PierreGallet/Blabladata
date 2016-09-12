# coding: utf8
from __future__ import print_function
import os, urllib, json, shutil, sys, time, csv, re, codecs, unicodedata, glob
from nltk.corpus import stopwords
import pandas as pd
from nltk.stem.snowball import FrenchStemmer
from datetime import datetime
from string import maketrans
from pos_tagging import lemmatize
from ner.predict import entities
from pprint import pprint


def parse(txt):
    """ We create the dictionary from the labeled data """
    # Input : Liste de 2-uple contenant (mail,label)
    # 0utput : Dictionnaire = liste de 2-uple avec l'ensemble des mots de tous les mails (mot,count)

    not_letters_or_digits =u'!"#%\'()*+,-./:;<=>?@[\]^_`{|}~1234567890'
    empty = u' '*len(not_letters_or_digits)
    trantab = maketrans(not_letters_or_digits, empty)
    stemmer = FrenchStemmer()

    a = txt.lower() # On met tout en minuscule
    a = a.replace('\n',' ') # On enleve les retours à la ligne
    a = a.replace('\r',' ') # On enleve les retours à la ligne
    a = a.translate(trantab) # On enleve la ponctuation
    a = a.decode('utf-8', 'replace')
    a = a.split() # On répartit selon les espaces
    #a = [stemmer.stem(word) for word in a]

    a = [word for word in a if not word.isdigit()]
    stop = stopwords.words('french')
    a = [word for word in a if not (word in stop)]  # On enleve les stopwords
    a = ' '.join(a)
    a = a.encode('utf-8', 'ignore')
    a = unicodedata.normalize('NFKD', unicode(a, 'utf-8')).encode('ASCII', 'ignore') # On enleve les accents et caractères speciaux
    a = ' '.join([word.strip(' \t\n\r') for word in a.split()])

    return a

def parse_soft(txt):
    """ We create the dictionary from the labeled data """
    # Input : Liste de 2-uple contenant (mail,label)
    # 0utput : Dictionnaire = liste de 2-uple avec l'ensemble des mots de tous les mails (mot,count)

    # not_letters_or_digits =u'!"#%\'()*+,-./:;<=>?@[\]^_`{|}~1234567890'
    # empty = u' '*len(not_letters_or_digits)
    # trantab = maketrans(not_letters_or_digits, empty)
    # stemmer = FrenchStemmer()

    a = txt.lower() # On met tout en minuscule
    a = a.replace('\n',' ') # On enleve les retours à la ligne
    a = a.replace('\r',' ') # On enleve les retours à la ligne
    # a = a.translate(trantab) # On enleve la ponctuation
    # a = a.decode('utf-8', 'replace')
    # a = a.split() # On répartit selon les espaces
    # # a = [stemmer.stem(word) for word in a]
    #
    # a = [word for word in a if not word.isdigit()]
    # stop = stopwords.words('french')
    # a = [word for word in a if not (word in stop)]  # On enleve les stopwords
    # a = ' '.join(a)
    # a = a.encode('utf-8', 'ignore')
    a = unicodedata.normalize('NFKD', unicode(a, 'utf-8')).encode('ASCII', 'ignore') # On enleve les accents et caractères speciaux
    a = ' '.join([word.strip(' \t\n\r') for word in a.split()])

    return a


class preprocessing():
    """
    lots of different methods to preprocess data
    """

    def __init__(self, data_directory, output_directory):
        """
        create the new directory (don't forget the path syntax : ./name_of_directory) and the input folder
        """
        self.output_directory = output_directory
        self.new_directory = './data/inputs'
        try:
            shutil.rmtree(self.new_directory)
        except:
            pass
        os.mkdir(self.new_directory)

        self.data_directory = data_directory

        self.path_sentences = self.new_directory+'/sentences.txt'
        self.path_labels = self.new_directory+'/labels.txt'

        # For tree_classifier
        self.path_labels_1 = self.new_directory+'/labels_1.txt'
        self.path_labels_2 = self.new_directory+'/labels_2.txt'
        self.path_labels_3 = self.new_directory+'/labels_3.txt'

        # for paraphrase detection
        self.path_sentences_1 = self.new_directory+'/sentences_1.txt'
        self.path_sentences_2 = self.new_directory+'/sentences_2.txt'

        try:
            self.raw_data = pd.read_csv(self.data_directory, sep=';')
        except:
            pass



    def label_indexing(self):
        """
        create a dictionary that maps word labels to indexes.
        """
        labels = self.raw_data.label.drop_duplicates().dropna()
        self.label_index = {}
        i = 0
        for label in labels:
            label = str(label)
            self.label_index[label] = i
            i += 1
        return self.label_index


    def get_number_of_classes(self):
        try:
            self.number_of_classes = len(self.label_index)
        except:
            self.number_of_classes = self.raw_data.groupby('label').size().size
        return self.number_of_classes


    def get_classes_names(self):
        target_names = list(self.raw_data.label.drop_duplicates().dropna())
        self.labels = {}
        for i in range(self.number_of_classes):
            self.labels[i] = target_names[i]

        if not os.path.exists(self.output_directory+'/models_saved'):
            makedirs(self.output_directory+'/models_saved')
        with open(self.output_directory+'/models_saved/classes.json', 'wb') as f:
            json.dump(self.labels, f, indent=4)

        return self.labels


    def csv(self, word_label=False):
        """
        works with a csv semi colon separated, with two field : label and sentence
        store the result in the new_directory/input section in .txt format
        """
        with open(self.path_sentences, 'w+') as sentences:
            with open(self.path_labels, 'w+') as labels:
                with open(self.data_directory, 'rb') as f:
                    reader = pd.read_csv(self.data_directory, sep=';')
                    if word_label == True:
                        label_index = self.label_indexing()
                    i = 0
                    for index,row in reader.iterrows():
                        if row['sentence']=='' or row['label']=='':
                            pass
                        else:
                            txt = parse(row['sentence'])
                            if word_label == True:
                                label = label_index[row['label']]
                            else:
                                label = row['label']
                            sentences.write(txt+'\n')
                            labels.write(str(label)+'\n')
        print('number of classes:', self.get_number_of_classes())
        print('classes names:')
        target_names = self.get_classes_names()  # here we save classes into the json
        pprint(target_names)
        print('...csv preprocessing ended')

    def csv_multi_motifs(self, word_label=False):
        """
        works with a csv semi colon separated, with two field : label and sentence
        store the result in the new_directory/input section in .txt format
        """
        with open(self.path_sentences, 'w+') as sentences:
            with open(self.path_labels, 'w+') as labels:
                with open(self.path_labels_1, 'w+') as labels_1:
                    with open(self.path_labels_2,'w+') as labels_2:
                        with open(self.path_labels_3,'w+') as labels_3:
                            with open(self.data_directory, 'rb') as f:
                                reader = csv.DictReader(f, fieldnames=['label', 'sentence','label_1','label_2','label_3'], delimiter=';')
                                i = 0
                                for row in reader:
                                    if i == 0:
                                        i += 1
                                    else:
                                        txt = parse(row['sentence'])
                                        label = row['label']
                                        label_1=row['label_1']
                                        label_2=row['label_2']
                                        label_3=row['label_3']
                                        sentences.write(txt+'\n')
                                        labels.write(str(label)+'\n')
                                        labels_1.write(str(label_1)+'\n')
                                        labels_2.write(str(label_2)+'\n')
                                        labels_3.write(str(label_3)+'\n')
            print('number of classes:', self.get_number_of_classes())
            print('classes names:')
            target_names = self.get_classes_names()  # here we save classes into the json
            pprint(target_names)
            print('...csv preprocessing ended')


    def txt_directory(self):
        """
        works with a directory where there is a folder for each labels, and in each of those folder, a number.txt for each sample, with number the N° of the sample.
        store the result in the new_directory/input section in .txt format
        """
        with open(self.path_sentences, 'w+') as sentences:
            with open(self.path_labels, 'w+') as labels:
                j = 0
                for directory in glob.glob(self.data_directory+'/*'):
                    files = os.listdir(directory)
                    print('dealing with', directory)
                    for i in range(len(files)):
                        with open(directory+'/'+files[i]) as f:
                            txt = parse(f.read())
                            if directory.split('/')[-1] == 'pos':
                                label = str(1)
                            else:
                                label = str(0)
                            sentences.write(txt+'\n')
                            labels.write(label+'\n')
                            j += 1
        print(str(j) + ' sentences preprocessed in total')
        return path_sentences, path_labels


    def paraphrase(self):
        """
        works with a directory where there is a folder for each labels, and in each of those folder, a number.txt for each sample, with number the N° of the sample.
        store the result in the new_directory/input section in .txt format
        """
        with open(self.path_sentences, 'w+') as sentences:
            with open(self.path_sentences_1, 'w+') as sentences_1:
                with open(self.path_sentences_2, 'w+') as sentences_2:
                    with open(self.path_labels, 'w+') as labels:
                        j = 0
                        with open(self.data_directory) as f:
                            for line in f.read().splitlines():
                                if (j % 3 == 0):
                                    labels.write(line+'\n')
                                elif (j % 3 == 1):
                                    sentences_1.write(parse(line)+'\n')
                                    sentences.write(parse(line)+'\n')
                                else:
                                    sentences_2.write(parse(line)+'\n')
                                    sentences.write(parse(line)+'\n')
                                j += 1


if __name__ == '__main__':

    word = "à l'école des beaÿ are"
    words = unicodedata.normalize('NFKD', unicode(word, 'utf-8')).encode('ASCII', 'ignore').split()
    print(words)
    print(parse_txt('écolier'))
