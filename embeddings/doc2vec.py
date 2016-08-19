#coding: utf8
# NOT WORKING
from __future__ import print_function
from gensim.models.doc2vec import Doc2Vec, TaggedLineDocument
from gensim.models import doc2vec
import logging
import numpy as np
from sklearn.cross_validation import train_test_split, StratifiedKFold
import pickle
from gensim import corpora, models
from collections import defaultdict
import operator

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class doc2vec():

    def __init__(self):
        """
        directory is the working directory where will be stored the templates & models & dictionary & others matrix
        """
        if not os.path.exists('./tmp'):
            os.makedirs('./tmp')

    def train(self, path_sentences, size=128, window=15, min_count=10):

        documents = TaggedLineDocument(path_sentences)
        model = Doc2Vec(documents, size=size, window=window, min_count=min)
        model.save('./tmp/doc2vec')
        print('...doc2vec training ended')
        return model

    def get_doc2vec_vectors(self, path_sentences):

        model = Doc2Vec.load('./tmp/doc2vec')
        emb_dim = model.vector_size
        with open(path_sentences, 'r+') as f:
            lines = f.readlines()
            sentences = np.zeros((len(lines), emb_dim))
            i = 0
            for line in lines:
                sentences[i] = model.docvecs[i]
                i += 1
        return sentences


    def format_input(self):

        try:
            shutil.rmtree('./data/inputs/doc2vec')
        except:
            pass
        os.mkdir('./data/inputs/doc2vec')

        self.path_sentences = './data/inputs/sentences.txt'
        self.path_labels = './data/inputs/labels.txt'
        self.path_sentences_output = './data/inputs/doc2vec/sentences.npy'
        self.path_labels_output = './data/inputs/doc2vec/labels.npy'

        sentences = self.get_doc2vec_vectors(self.path_sentences)

        with open(self.path_labels, 'r+') as f:
            lines = f.readlines()
            labels = np.zeros((len(lines), 1))
            i = 0
            for line in lines:
                labels[i] = line
                i += 1

        print("saving formated input")
        with open(self.path_sentences_output, 'wb') as f:
            np.save(f, sentences)
        with open(self.path_labels_output, 'wb') as f:
            np.save(f, labels)

        print('shape of sentences (nb_sample, max_len):', sentences.shape)
        print('shape of labels (nb_sample):', labels.shape)
        return sentences, labels


if __name__ == '__main__':

    path_sentences='./data/inputs/sentences.txt'
    model = doc2vec()
    model = model.train(path_sentences)
    print(model.most_similar(positive=['forfait']))
