# coding: utf8
from __future__ import print_function
import os, urllib, json, shutil, sys, time, csv, re, codecs, unicodedata, glob
from nltk.corpus import stopwords
import pandas as pd


class prepocessing():
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
        stopwordsfile = stopwords.words('english')
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
            # remove stopwords
            if word and (word not in stop_words_set) and (len(word) > 1):
                try:
                    words = unicodedata.normalize('NFKD', unicode(word, 'utf-8')).encode('ASCII', 'ignore').split()
                    for word in words:
                        if word and (word not in stop_words_set) and (len(word) > 1):
                            clean_txt = clean_txt + word + ' '
                except:
                    pass
        return clean_txt


    def label_indexing(self):
        """
        create a new csv with numerical label instead of words
        """
        raw_data = pd.read_csv(self.data_directory, sep=';')
        labels = raw_data.label.drop_duplicates()
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

    ### for Deep learning ####
    # datadir = './data/Stanford - IMDB review sentiment analysis dataset/ang/test'

    # os.mkdir('./dl')
    # os.mkdir('./dl/tmp')
    # os.mkdir('./dl/tmp/models_saved')
    # os.mkdir('./dl/input')
    # os.mkdir('./dl/input/test')
    # os.mkdir('./dl/input/formated')
    # os.mkdir('./dl/input/formated/test')

    # path_sentences = './dl/input/test/sentences.txt'
    # path_labels = './dl/input/test/labels.txt'


    # #### for Machine learning ####
    # datadir = './data/kaggle - Bag of Words Meets Bags of Popcorn/train.csv'
    #
    # os.mkdir('./ml')
    # os.mkdir('./ml/tmp')
    # os.mkdir('./ml/tmp/models_saved')
    # os.mkdir('./ml/input')
    # os.mkdir('./ml/input/formated')
    #
    # path_sentences = './ml/input/sentences.txt'
    # path_labels = './ml/input/labels.txt'

    ### for Paraphrase detection ###
    datadir = './data/MRPC/train.txt'
    # try:
    #     shutil.rmtree('./pd')
    # except:
    #     pass
    # os.mkdir('./pd')
    # os.mkdir('./pd/tmp')
    # os.mkdir('./pd/tmp/models_saved')
    # os.mkdir('./pd/input')
    # os.mkdir('./pd/input/train')
    # os.mkdir('./pd/input/test')
    # os.mkdir('./pd/input/formated')
    # os.mkdir('./pd/input/formated/train')
    # os.mkdir('./pd/input/formated/test')

    path_sentences = './pd/input/train/sentences.txt'
    path_sentences_1 = './pd/input/train/sentences_1.txt'
    path_sentences_2 = './pd/input/train/sentences_2.txt'
    path_labels = './pd/input/train/labels.txt'

    ################################
    # if datadir.split('.')[-1] == 'csv':
    #     preprocessing_csv(datadir, path_sentences, path_labels)
    # else:
    #     preprocessing_diroftxt(datadir, path_sentences, path_labels)

    preprocessing_paraphrase(path_sentences, path_sentences_1, path_sentences_2, path_labels)
