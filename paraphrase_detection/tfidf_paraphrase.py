# coding: utf8
from __future__ import print_function
from gensim.models.word2vec import Word2Vec, LineSentence
from gensim.models import word2vec
import logging
import numpy as np
from sklearn.cross_validation import train_test_split, StratifiedKFold
from gensim.models.word2vec import Word2Vec, LineSentence
import pickle, json, os
from gensim import corpora, models
from collections import defaultdict
import operator
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
np.set_printoptions(suppress=True)
np.set_printoptions(threshold='nan')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class tfidf_paraphrase():

    def __init__(self, directory):
        self.directory = directory
        if not os.path.exists('./paraphrase_detection/tmp'):
            os.makedirs('./paraphrase_detection/tmp')
        if not os.path.exists(self.directory+'/input/tfidf'):
            os.makedirs(self.directory+'/input/tfidf')

    def train(self, path_sentences):
        """
        use sklearn to compute tfidf weight vectors
        """
        def myCorpus():
            for line in open(path_sentences):
                yield line
        corpus = myCorpus()

        vectorizer = TfidfVectorizer(encoding='utf8')
        vectorizer.fit(corpus)

        with open('./paraphrase_detection/tmp/tfidf.pk', 'wb') as f:
            pickle.dump(vectorizer, f)

        print('...tfidf computation ended')


    def feature_selection(self, sentences, threshold=0.7):
        max_tfidf = np.amax(sentences, axis=0)
        list = sorted(max_tfidf.tolist())
        print('tfidf inferior to the', int(threshold * len(list)), 'element in list, which is', list[int(threshold * len(list))-1], 'will be removed')
        index_to_delete = []
        for i in range(len(list)):
            if max_tfidf[i] < list[int(threshold * len(list))-1]:
                index_to_delete.append(i)
        sentences = np.delete(sentences, index_to_delete, axis=1)
        return sentences


    def format_input(self, feature_selection_threshold=0.85):
        self.path_sentences_1 = self.directory + '/input/sentences_1.txt'
        self.path_sentences_2 = self.directory + '/input/sentences_2.txt'
        self.path_labels = self.directory + '/input/labels.txt'
        self.path_sentences_output = self.directory + '/input/tfidf/sentences.npy'
        self.path_labels_output = self.directory + '/input/tfidf/labels.npy'

        # we load the tfidf vectorizer
        with open('./paraphrase_detection/tmp/tfidf.pk', 'rb') as f:
            vectorizer = pickle.load(f)

        # we load the sentences_1 data
        def myCorpus_1():
            for line in open(self.path_sentences_1):
                yield line
        corpus_1 = myCorpus_1()
        # vectorization + densification of the sparse matrix obtained
        sentences_1 = vectorizer.transform(corpus_1)
        sentences_1 = sentences_1.todense()
        sentences_1 = np.asarray(sentences_1)  # size samples x lenght of vocabulary

        # print("shape before feature selection:", sentences_1.shape)
        # print("feature selection starting")
        # sentences_1 = self.feature_selection(sentences_1, threshold=feature_selection_threshold)
        # print("shape after feature selection:", sentences_1.shape)

        # we load the sentences_2 data
        def myCorpus_2():
            for line in open(self.path_sentences_2):
                yield line
        corpus_2 = myCorpus_2()
        # vectorization + densification of the sparse matrix obtained
        sentences_2 = vectorizer.transform(corpus_2)
        sentences_2 = sentences_2.todense()
        sentences_2 = np.asarray(sentences_2)  # size samples x lenght of vocabulary

        print("shape before feature selection:", sentences_2.shape)
        print("feature selection starting")
        sentences_2 = self.feature_selection(sentences_2, threshold=feature_selection_threshold)
        print("shape after feature selection:", sentences_2.shape)

        sentences = np.concatenate((sentences_1, sentences_2), axis=1)

        with open(self.path_labels, 'r+') as f:
            lines = f.readlines()
            labels = np.zeros((len(lines), 1))
            i = 0
            for line in lines:
                labels[i] = int(line)
                i += 1

        print("saving formated input")
        with open(self.path_sentences_output, 'wb') as f:
            np.save(f, sentences)
        with open(self.path_labels_output, 'wb') as f:
            np.save(f, labels)

        print('shape of sentences', sentences.shape)
        print('shape of labels', labels.shape)
        return sentences, labels


    # def get_tfidf_vectors_gensim(path_sentences):
    #
    #     def myCorpus():
    #         for line in open(path_sentences):
    #             yield line.lower()
    #
    #     # instantiate the corpus
    #     corpus = myCorpus()
    #     # create the dictionary
    #     dictionary = corpora.Dictionary(corpus)
    #     # prune the dictionary
    #     dictionary.filter_extremes(no_below=5, no_above=0.5)
    #     # compute the bag of words for each sentence
    #     bow = [dictionary.doc2bow(text) for text in corpus]
    #     # initialize tfidf model
    #     tfidf = models.TfidfModel(bow)
    #     # tfidf.save('./ml/tmp/tfidf.tfidf')
    #     # apply to corpus
    #     # corpus_tfidf = tfidf[bow]
    #     sentences = np.zeros((len(bow), len(dictionary)))  # size = samples x lenght of vocabulary
    #     print("i arrived there")
    #     i = 0
    #     for vector in bow:
    #         for j in range(len(vector)):
    #             sentences[i, tfidf[vector][j][0]] = tfidf[vector][j][1]
    #         i += 1
    #     print("tfidf finished")
    #     return sentences

if __name__ == '__main__':

    path_sentences = './ml/input/sentences.txt'
    path_labels = './ml/input/labels.txt'

    sentences, labels = formatting_ml_input(path_sentences, path_labels)
    X_train, X_val, y_train, y_val = load_data()
    print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
