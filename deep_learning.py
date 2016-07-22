#coding: utf8
'''
Train a recurrent convolutional network on the IMDB sentiment
classification task.
Gets to 0.8498 test accuracy after 2 epochs. 41s/epoch on K520 GPU.
'''
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM, GRU, SimpleRNN
from keras.layers import Convolution1D, MaxPooling1D, AveragePooling2D, AveragePooling1D
from keras.datasets import imdb
from keras.utils.np_utils import accuracy
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split, StratifiedKFold
import pickle, time
import matplotlib.pyplot as plt
import os
np.set_printoptions(suppress=True)
np.set_printoptions(threshold='nan')

class deep_learning():

    def __init__(self, directory):
        self.directory = directory
        self.path_sentences = directory + '/input/word2vec/sentences.npy'
        self.path_labels = directory + '/input/word2vec/labels.npy'

    def prepare_data(self, test_size=0.20, max_len=150):
        print('...Preparing data...')
        with open(self.path_sentences, 'rb') as sentences_npy:
            with open(self.path_labels, 'rb') as labels_npy:
                sentences = np.array(np.load(sentences_npy))
                labels = np.array(np.load(labels_npy))
                X_train, X_val, y_train, y_val = train_test_split(sentences, labels, test_size=test_size, random_state=1000)
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        print(len(self.X_train), 'train sequences')
        print(len(self.X_val), 'validation sequences')

        print('...Pad sequences...')
        self.X_train = sequence.pad_sequences(self.X_train, maxlen=max_len, truncating='post')
        self.X_val = sequence.pad_sequences(self.X_val, maxlen=max_len, truncating='post')
        # self.X_test = sequence.pad_sequences(X_test, maxlen=max_len, truncating='post')
        print('X_train shape:', self.X_train.shape)  # should be = samples x maxlen
        print('X_val shape:', self.X_val.shape)
        print(self.X_train[:5], self.y_train[:5])

    # def load_data_doc2vec():
    #     print("loading data")
    #     with open('./ml/input/formated//doc2vec/sentences.npy', 'rb') as sentences_npy:
    #         with open('./ml/input/formated/doc2vec/labels.npy', 'rb') as labels_npy:
    #             sentences = np.array(np.load(sentences_npy))
    #             labels = np.array(np.load(labels_npy))
    #             X_train, X_val, y_train, y_val = train_test_split(sentences, labels, test_size=0.20, random_state=100)
    #     return X_train, X_val, y_train, y_val


    def build_lstm_cnn(self, max_len=100, filter_length=3, nb_filter=64, pool_length=2, lstm_output_size=70, number_of_classes=2):
        print('...Build model...')
        with open(self.directory + '/tmp/embedding_weights.pk', 'rb') as weights_pk:
            self.embedding_weights = np.array(pickle.load(weights_pk))   # size = voc_dim x emb_dim
            # print(weights[:10])
        self.voc_dim, self.emb_dim = self.embedding_weights.shape

        self.model = Sequential()
        self.model.add(Embedding(self.voc_dim, self.emb_dim, input_length=max_len, weights=[self.embedding_weights]))
        self.model.add(Dropout(0.25))
        self.model.add(Convolution1D(nb_filter=nb_filter,
                                     filter_length=filter_length,
                                     border_mode='valid',
                                     activation='relu',
                                     subsample_length=1))
        self.model.add(MaxPooling1D(pool_length=pool_length))
        self.model.add(LSTM(lstm_output_size, return_sequences=False, consume_less="cpu"))
        self.model.add(Dense(number_of_classes))
        self.model.add(Activation('softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    def build_simple_nn(self):
        print('...Build model...')

        self.model = Sequential()
        self.model.add(Dense(1000, input_dim=10000, init='normal', activation='relu'))
        self.model.add(Dense(100, input_dim=1000, init='normal', activation='relu'))
        self.model.add(Dense(10, input_dim=100, init='normal', activation='relu'))
        self.model.add(Dense(1, init='normal', activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


    def train(self, batch_size=30, nb_epoch=5):
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        print('...Train...')
        start_time = time.time()
        self.history = self.model.fit(self.X_train, self.y_train, batch_size=self.batch_size, nb_epoch=self.nb_epoch, validation_data=(self.X_val, self.y_val), verbose=2)
        self.average_time_per_epoch = (time.time() - start_time) / self.nb_epoch

        print('...Saving model...')

        if not os.path.exists(self.directory+'/tmp/models_saved'):
            os.makedirs(self.directory+'/tmp/models_saved')

        try:
            print('starting saving weights')
            self.model.save_weights(self.directory + '/tmp/models_saved/lstm_cnn_weights.h5')
            print('ending saving weights')
            model_saved = self.model.to_json()
            with open(self.directory + '/tmp/models_saved/lstm_cnn.json', 'w+') as f:
                print('starting saving model')
                f.write(model_saved)
                print('ending saving model')
        except:
            print('could not save the model')
            pass


    def predict(self):
        # compute prediction on validation set
        self.pred = self.model.predict(self.X_val, batch_size=self.batch_size)

        # transform the prediction matrix into a vector with labels indexes
        print(self.pred[:5])
        self.pred_vector = np.zeros(self.pred.shape[0])
        i = 0
        for k in list(np.argmax(np.array(self.pred, dtype=float), axis=1)):
            self.pred_vector[i] = k
            i += 1
        print(self.pred_vector)

        # idem for y_val matrix
        print(self.y_val[:5])
        self.y_val_vector = np.zeros(self.y_val.shape[0])
        i = 0
        for k in list(np.argmax(np.array(self.y_val, dtype=float), axis=1)):
            self.y_val_vector[i] = k
            i += 1
        print(self.y_val_vector)

        self.accuracy = accuracy_score(self.y_val_vector, self.pred_vector)
        self.confusion_matrix = np.array(confusion_matrix(self.y_val_vector, self.pred_vector), dtype=float)

        # self.ratio = {}
        # self.ratio['false_pos'] = (self.confusion_matrix[0, 1]/np.sum(self.confusion_matrix[0, :]))
        # self.ratio['false_neg'] = (self.confusion_matrix[1, 0]/np.sum(self.confusion_matrix[1, :]))
        # self.ratio['true_pos'] = (self.confusion_matrix[1, 1]/np.sum(self.confusion_matrix[1, :]))
        # self.ratio['true_neg'] = (self.confusion_matrix[0, 0]/np.sum(self.confusion_matrix[0, :]))

        print('\n\n## FINISHED ##')
        print('\nresult for the cnn + lstm on validation set:')
        print('accuracy:', self.accuracy, '\nconfusion matrix:\n', self.confusion_matrix, '\naverage time per epoch:', self.average_time_per_epoch)
        print('\nadditional metrics:')
        print('history:', self.history.history)


    def test(self):
        #TODO: rendre fonctionnelle la partie test sur unseen data
        self.pred_test = np.round(model.predict(self.X_test, batch_size=batch_size))
        self.accuracy_test = accuracy(self.pred_test, self.y_test)
        self.confusion_matrix_test = np.array(confusion_matrix(self.y_test, self.pred_test), dtype=float)
        self.ratio_test = {}
        self.ratio_test['false_pos'] = (self.confusion_matrix_test[0, 1]/np.sum(self.confusion_matrix_test[0, :]))
        self.ratio_test['false_neg'] = (self.confusion_matrix_test[1, 0]/np.sum(self.confusion_matrix_test[1, :]))
        self.ratio_test['true_pos'] = (self.confusion_matrix_test[1, 1]/np.sum(self.confusion_matrix_test[1, :]))
        self.ratio_test['true_neg'] = (self.confusion_matrix_test[0, 0]/np.sum(self.confusion_matrix_test[0, :]))

        print('\nresult for the cnn + lstm on test set:')
        print('accuracy:', self.accuracy_test, '\nconfusion matrix:\n', self.confusion_matrix_test)
        print('\nadditional metrics:')
        print('ratio', self.ratio_test)


    def get_plots(self, title='plot'):
        # show plot`
        plt.plot(list(range(self.nb_epoch)), self.history.history['acc'], label='training accuracy')
        plt.plot(list(range(self.nb_epoch)), self.history.history['loss'], label='training loss')
        plt.plot(list(range(self.nb_epoch)), self.history.history['val_acc'], label='validation accuracy')
        plt.plot(list(range(self.nb_epoch)), self.history.history['val_loss'], label='validation loss')
        plt.title(title)
        plt.legend(loc="lower right", prop={'size': 6})
        plt.xlabel('epoch')
        plt.ylabel('accuracy/loss')
        plt.show()


if __name__ == '__main__':

    #INPUTS
    path_sentences = './dl/input/formated/train/sentences.npy'
    path_labels = './dl/input/formated/train/labels.npy'

    path_sentences_test = './dl/input/formated/test/sentences.npy'
    path_labels_test = './dl/input/formated/test/labels.npy'

    # Embedding
    voc_dim = 19376  # also called max_feature
    max_len = 100
    emb_size = 128

    # Convolution
    filter_length = 3
    nb_filter = 64
    pool_length = 2

    # LSTM
    lstm_output_size = 70

    # Training
    batch_size = 30  # batch_size is highly sensitive
    nb_epoch = 5  # Only 2 epochs are needed as the dataset is very small.

    ## FOR CNN + LSTM ###
    X_train, X_val, y_train, y_val, X_test, y_test = prepare_data()
    model = build_lstm_cnn()
    model, history, average_time_per_epoch, score, acc, conf, ratio = train(model, X_train, X_val, y_train, y_val)
    acc_test, conf_test, ratio_test = test(model, X_test, y_test)

    ### FOR SIMPLE NN ###
    # X_train, X_val, y_train, y_val = prepare_data()
    # model = build_simple_nn()
    # model, history, average_time_per_epoch, score, acc, conf, ratio = train(model, X_train, X_val, y_train, y_val)

    print('\n\n## FINISHED ##')
    print('\nresult for the cnn + lstm on train:')
    print('accuracy:', acc, '\nconfusion matrix:\n', conf, '\naverage time per epoch:', average_time_per_epoch)
    print('\nadditional metrics:')
    print('history:', history.history, '\nscore', score, '\nratio', ratio)

    print('\nresult for the cnn + lstm on test:')
    print('accuracy:', acc_test, '\nconfusion matrix:\n', conf_test)
    print('\nadditional metrics:')
    print('ratio', ratio_test)

    # show plot`
    plt.plot(list(range(nb_epoch)), history.history['acc'], label='training accuracy')
    plt.plot(list(range(nb_epoch)), history.history['loss'], label='training loss')
    plt.plot(list(range(nb_epoch)), history.history['val_acc'], label='validation accuracy')
    plt.plot(list(range(nb_epoch)), history.history['val_loss'], label='validation loss')
    plt.title('CNN + LSTM results')
    plt.legend(loc="lower right", prop={'size': 6})
    plt.xlabel('epoch')
    plt.ylabel('accuracy/loss')
    plt.show()
