# coding: utf8
from __future__ import print_function
import os, urllib, json, shutil, sys, time, csv, re, codecs, unicodedata, glob
from nltk.corpus import stopwords
import pandas as pd
from nltk.stem.snowball import FrenchStemmer
from datetime import datetime
from string import maketrans

def parse_txt(txt):
    """
    Splits a string into an lean string without punctiation, carriage returns & stopwords.
    """

    punctuation = ['(', ')', ':', ';', '«', '»', ',', '-', '!', '.', '?', '/', '[', ']', '{', '}',
                   '#', '"', '*', '-', "`", '"', '>>', '|', '/', '*', '•', ' ', "d'", "j'",
                   "t'", "l'", "s'", "n'", "qu'", "c'"]

    carriage_returns = ['\n', '\r\n']

    word_regex = "^[a-zàâçéèêëîïôûùüÿñæœ/+ .-]+$"

    stop_words_set = set()
    stopwordsfile = stopwords.words('french')
    for word in stopwordsfile:  # a stop word in each line
        word = word.replace("\n", '')
        word = word.replace("\r\n", '')
        stop_words_set.add(word)

    clean_txt = ''
    words = txt.split()
    for word in words:
        # lower case
        word = word.lower()
        # remove punctuation & carriage returns
        for punc in punctuation + carriage_returns:
            word = word.replace(punc, ' ').strip(' ')
        # check if it is normal letters
        if not re.match(word_regex, word):
            word = None
        # stemming
        stemmer = FrenchStemmer()
        # word = stemmer.stem(word)
        # remove stopwords
        if word and (word not in stop_words_set) and (len(word) > 1):
            try:
                words = unicodedata.normalize('NFKD', unicode(word, 'utf-8')).encode('ASCII', 'ignore').split()
                for word in words:
                    if word and (word not in stop_words_set) and (len(word) > 1):
                        clean_txt = clean_txt + stemmer.stem(word) + ' '
            except:
                pass
    return clean_txt


class preprocessing():
    """
    lots of different methods to preprocess data
    """

    def __init__(self, data_directory, new_directory):
        """
        create the new directory (don't forget the path syntax : ./name_of_directory) and the input folder
        """
        self.new_directory = new_directory
        try:
            shutil.rmtree(self.new_directory)
        except:
            pass
        os.mkdir(self.new_directory)

        self.data_directory = data_directory

        os.mkdir(self.new_directory+'/input')
        self.path_sentences = self.new_directory+'/input/sentences.txt'
        self.path_labels = self.new_directory+'/input/labels.txt'

        # for paraphrase detection
        self.path_sentences_1 = self.new_directory+'/input/sentences_1.txt'
        self.path_sentences_2 = self.new_directory+'/input/sentences_2.txt'

    def parse_txt(self, txt):
        """
        Splits a string into an lean string without punctiation, carriage returns & stopwords.
        """

        punctuation = ['(', ')', ':', ';', '«', '»', ',', '-', '!', '.', '?', '/', '[', ']', '{', '}',
                       '#', '"', '*', '-', "`", '"', '>>', '|', '/', '*', '•', ' ', "d'", "j'",
                       "t'", "l'", "s'", "n'", "qu'", "c'"]

        carriage_returns = ['\n', '\r\n']

        word_regex = "^[a-zàâçéèêëîïôûùüÿñæœ/+ .-]+$"

        stop_words_set = set()
        stopwordsfile = stopwords.words('french')
        for word in stopwordsfile:  # a stop word in each line
            word = word.replace("\n", '')
            word = word.replace("\r\n", '')
            stop_words_set.add(word)

        clean_txt = ''
        words = txt.split()
        for word in words:
            # lower case
            word = word.lower()
            # remove punctuation & carriage returns
            for punc in punctuation + carriage_returns:
                word = word.replace(punc, ' ').strip(' ')
            # check if it is normal letters
            if not re.match(word_regex, word):
                word = None
            # stemming
            stemmer = FrenchStemmer()
            # word = stemmer.stem(word)
            # remove stopwords
            if word and (word not in stop_words_set) and (len(word) > 1):
                try:
                    words = unicodedata.normalize('NFKD', unicode(word, 'utf-8')).encode('ASCII', 'ignore').split()
                    for word in words:
                        if word and (word not in stop_words_set) and (len(word) > 1):
                            clean_txt = clean_txt + stemmer.stem(word) + ' '
                except:
                    pass
        return clean_txt

    def parse_max (self,txt):
        """ We create the dictionary from the labeled data """
        # Input : Liste de 2-uple contenant (mail,label)
        # 0utput : Dictionnaire = liste de 2-uple avec l'ensemble des mots de tous les mails (mot,count)

        not_letters_or_digits =u'!"#%\'()*+,-./:;<=>?@[\]^_`{|}~1234567890'
        empty = u' '*len(not_letters_or_digits)
        trantab = maketrans(not_letters_or_digits,empty)
        stemmer = FrenchStemmer()

        a = txt.replace('\n',' ')
        a = txt.translate(trantab) # On enleve la ponctuation
        a = a.lower() # On met tout en minuscule
        a = a.decode('utf-8', 'replace')
        a = a.split() # On répartit selon les espaces
        a = [stemmer.stem(word) for word in a]

        a = [word for word in a if not word.isdigit()]
        stop = stopwords.words('french')
        a = [word for word in a if not (word in stop)]

        a= ' '.join(a)
        a = a.encode('utf-8', 'ignore')

        return a

    def label_indexing(self):
        """
        create a new csv with numerical label instead of words
        """
        raw_data = pd.read_csv(self.data_directory, sep=';')
        labels = raw_data.label.drop_duplicates().dropna()
        self.label_index = {}
        i = 0
        for label in labels:
            self.label_index[label] = i
            i += 1
        print('number of classes:', len(self.label_index))
        return self.label_index


    def get_number_of_classes(self):
        return len(self.label_index)


    def get_classes_names(self):
        raw_data = pd.read_csv(self.data_directory, sep=';')
        labels = list(raw_data.label.drop_duplicates().dropna())
        return labels


    def csv(self, word_label=False):
        """
        works with a csv semi colon separated, with two field : label and sentence
        store the result in the new_directory/input section in .txt format
        """
        with open(self.path_sentences, 'w+') as sentences:
            with open(self.path_labels, 'w+') as labels:
                with open(self.data_directory, 'rb') as f:
                    reader = csv.DictReader(f, fieldnames=['label', 'sentence'], delimiter=';')
                    if word_label == True:
                        label_index = self.label_indexing()
                    i = 0
                    for row in reader:
                        if i == 0:
                            i += 1
                        else:
                            txt = self.parse_txt(row['sentence'])
                            if word_label == True:
                                label = label_index[row['label']]
                            else:
                                label = row['label']
                            sentences.write(txt+'\n')
                            labels.write(str(label)+'\n')
        print('...csv preprocessing ended')

    def csv2(self, word_label=False):
        """
        works with a csv semi colon separated, with two field : label and sentence
        store the result in the new_directory/input section in .txt format
        """
        with open(self.path_sentences, 'w+') as sentences:
            with open(self.path_labels, 'w+') as labels:
                with open(self.data_directory, 'rb') as f:
                    reader = csv.DictReader(f, fieldnames=['label', 'sentence', 'a','b'], delimiter=';')
                    if word_label == True:
                        label_index = self.label_indexing()
                    i = 0
                    for row in reader:
                        if row['sentence']=='' or row['label']=='':
                            pass
                        else:
                            if i == 0:
                                i += 1
                            else:
                                txt = self.parse_max(row['sentence'])
                                if word_label == True:
                                    print(i)
                                    print(pd.isnull(row['label']))
                                    print(row['label'])
                                    label = label_index[row['label']]
                                    i+=1
                                else:
                                    label = row['label']
                                sentences.write(txt+'\n')
                                labels.write(str(label)+'\n')
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
                            txt = self.parse_txt(f.read())
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
                                    sentences_1.write(self.parse_txt(line)+'\n')
                                    sentences.write(self.parse_txt(line)+'\n')
                                else:
                                    sentences_2.write(self.parse_txt(line)+'\n')
                                    sentences.write(self.parse_txt(line)+'\n')
                                j += 1


if __name__ == '__main__':

    word = "à l'école des beaÿ are"
    words = unicodedata.normalize('NFKD', unicode(word, 'utf-8')).encode('ASCII', 'ignore').split()
    print(words)
    print(parse_txt('écolier'))
