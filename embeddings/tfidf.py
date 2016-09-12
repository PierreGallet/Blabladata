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


class tfidf():

    def __init__(self, output_directory):
        self.output_directory = output_directory
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        if not os.path.exists('./data/inputs/tfidf'):
            os.makedirs('./data/inputs/tfidf')

    def train(self, path_sentences):
        """
        use sklearn to compute tfidf weight vectors
        """
        def myCorpus():
            for line in open(path_sentences):
                yield line
        corpus = myCorpus()

        # Plus tard toucher à ngram_range pour avoir aussi des bi-grams
        vectorizer = TfidfVectorizer(encoding='utf8')
        #vectorizer = TfidfVectorizer(encoding='utf8',ngram_range=(2,2),min_df=3)
        vectorizer.fit(corpus)

        with open(self.output_directory + '/tfidf.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)

        print('...train tfidf ended')


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
        self.path_sentences = './data/inputs/sentences.txt'
        self.path_labels = './data/inputs/labels.txt'
        self.path_sentences_output = './data/inputs/tfidf/sentences.npy'
        self.path_labels_output = './data/inputs/tfidf/labels.npy'

        # we load the tfidf vectorizer
        with open(self.output_directory + '/tfidf.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        # we load the sentences data
        def myCorpus():
            for line in open(self.path_sentences):
                yield line
        corpus = myCorpus()
        # vectorization + densification of the sparse matrix obtained
        sentences = vectorizer.transform(corpus)
        sentences = sentences.todense()
        sentences = np.asarray(sentences)  # size samples x lenght of vocabulary

        # print("shape before feature selection:", sentences.shape)
        # print("feature selection starting")
        # sentences = self.feature_selection(sentences, threshold=feature_selection_threshold)
        # print("shape after feature selection:", sentences.shape)

        with open(self.path_labels, 'r+') as f:
            lines = f.readlines()
            labels = np.zeros((len(lines), 1))
            i = 0
            for line in lines:
                labels[i] = int(line)
                i += 1

        ### WAY TOO LONG FOR BIG DATA (so we don)
        # print("saving formated input")
        # with open(self.path_sentences_output, 'wb') as f:
        #     np.save(f, sentences)
        # with open(self.path_labels_output, 'wb') as f:
        #     np.save(f, labels)



        print('shape of sentences', sentences.shape)
        print('shape of labels', labels.shape)
        return sentences, labels

    def format_input_tree(self, feature_selection_threshold=0.85):
        self.path_sentences = './data/inputs/sentences.txt'
        self.path_labels = './data/inputs/labels.txt'
        self.path_labels_1 = './data/inputs/labels_1.txt'
        self.path_labels_2 = './data/inputs/labels_2.txt'
        self.path_labels_3 = './data/inputs/labels_3.txt'

        self.path_sentences_output = './data/inputs/tfidf/sentences.npy'
        self.path_labels_output = './data/inputs/tfidf/labels.npy'
        self.path_labels_output_1='./data/inputs/tfidf/labels_1.npy'
        self.path_labels_output_2='./data/inputs/tfidf/labels_2.npy'
        self.path_labels_output_3='./data/inputs/tfidf/labels_3.npy'

        # we load the tfidf vectorizer
        with open(self.output_directory + '/tfidf.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        # we load the sentences data
        def myCorpus():
            for line in open(self.path_sentences):
                yield line
        corpus = myCorpus()
        # vectorization + densification of the sparse matrix obtained
        sentences = vectorizer.transform(corpus)
        sentences = sentences.todense()
        sentences = np.asarray(sentences)  # size samples x lenght of vocabulary

        with open(self.path_sentences_output, 'wb') as f:
            np.save(f, sentences)
        print('shape of sentences', sentences.shape)

        # print("shape before feature selection:", sentences.shape)
        # print("feature selection starting")
        # sentences = self.feature_selection(sentences, threshold=feature_selection_threshold)
        # print("shape after feature selection:", sentences.shape)

        def create_np_labels(fichier_txt,fichier_np):
            with open(fichier_txt, 'r+') as f:
                lines = f.readlines()
                name_np = np.zeros((len(lines), 1))
                i = 0
                for line in lines:
                    try:
                        name_np[i] = int(line)
                        i += 1
                    except ValueError:
                        name_np[i] = np.nan
                        i +=1
            with open(fichier_np, 'wb') as f:
                np.save(f, name_np)
            print('shape of labels', name_np.shape)

        #labels = create_np_labels(self.path_labels,self.path_labels_output)
        labels_1 = create_np_labels(self.path_labels_1,self.path_labels_output_1)
        labels_2 = create_np_labels(self.path_labels_2,self.path_labels_output_2)
        labels_3 = create_np_labels(self.path_labels_3,self.path_labels_output_3)

        #return sentences, labels

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

    path_sentences = './data/inputs/sentences.txt'
    model = tfidf()
    model = model.train(path_sentences)
    model
