# coding: utf8
import pickle, operator
import numpy as np
import formating
import preprocessing
import word2vec
import deep_learning
import machine_learning
import tfidf
from sklearn.metrics.pairwise import cosine_similarity
# np.set_printoptions(threshold='nan')
# np.set_printoptions(suppress=True)

deep = False
# dictionary that links machine learning models to their parameters
ml_models = {}
# ml_models['reglog_l1'] = 1.0  # C
ml_models['reglog_l2'] = 1.0  # C
# ml_models['reglog_sgd'] = 0.0001  # alpha
# ml_models['naive_bayes'] = ''
# ml_models['decision_tree'] = 'gini'  # entropy
# ml_models['random_forest'] = 5  # nb_estimator
# ml_models['bagging_reglog_l1'] = 5  # nb_estimator
# ml_models['bagging_reglog_l2'] = 5  # nb_estimator
# ml_models['svm_linear'] = 1.0  # C
# ml_models['knn'] = 5  # nb_neighbors


# the inputs
data_directory = './data/SFR/messages_formated_cat.csv'
new_directory = './sfr'

# go from raw_data to a csv with 2 column sentence/label
# formating.formating_csv('./data/SFR/messages.csv')

# preprocess the data within input/sentence.txt and input/label.txt
preprocessing = preprocessing.prepocessing(data_directory, new_directory)
preprocessing.csv2(word_label=True)
number_of_classes = preprocessing.get_number_of_classes()

if deep is True:
    # compute word2vec and assign the right index to each word to make a proper numerical matrix to feed our model.
    word2vec = word2vec.word2vec(new_directory)
    word2vec.train(new_directory+'/input/sentences.txt', size=128)
    word2vec.format_input()

    # the models
    dl = deep_learning.deep_learning(new_directory)
    dl.prepare_data(test_size=0.20, max_len=150)
    dl.build_lstm_cnn(max_len=150, filter_length=3, nb_filter=64, pool_length=2, lstm_output_size=70, number_of_classes=number_of_classes)
    dl.train(batch_size=30, nb_epoch=2)
    target_name = preprocessing.get_classes_names()
    dl.predict(target_name)
    dl.get_plots()

else:
    # compute tfidf and transform the sentences into tfidf vectors, with feature selection (best tfidf weights)
    tfidf = tfidf.tfidf(new_directory)
    tfidf.train(new_directory+'/input/sentences.txt')
    tfidf.format_input(feature_selection_threshold=0.85)

    # the models
    ml = machine_learning.machine_learning(new_directory)
    ml.prepare_data(test_size=0.20)
    for model_name, params in ml_models.items():
        ml.build(model_name, params)
        ml.train()
        target_name = preprocessing.get_classes_names()
        ml.predict(target_name)


    # # test cosinus similarité
    # with open('./sfr/input/tfidf/sentences.npy', 'rb') as sent:
    #     sentences = np.array(np.load(sent))
    #
    # test_sentences = sentences[19]
    # print test_sentences
    # results = {}
    # print sentences.shape[0]
    # for k in range(sentences.shape[0]):
    #     results[k] = float(cosine_similarity(test_sentences, sentences[k]))
    # with open('./sfr/input/sentences.txt', 'rb') as text:
    #     sentences = text.readlines()
    #     print 'finding similar sentences to :', sentences[19]
    #     for i, cosine_sim in sorted(results.items(), key=operator.itemgetter(1), reverse=True)[:10]:
    #         print sentences[i]
