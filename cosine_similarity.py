# coding: utf8
import pickle, operator, os, json, time
import numpy as np
import formating
import preprocessing
import word2vec
import deep_learning
import machine_learning
import tfidf
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
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
data_directory = './data/SFR/messages_formated_QA.csv'
new_directory = './sfr'

def vectorize_qa():
    with open('./tmp/tfidf.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    raw_data = pd.read_csv('./data/SFR/messages_formated_QA.csv', sep=';')
    def myCorpus():
        for sentence in raw_data['question']:
            yield preprocessing.parse_txt(sentence)
    corpus = myCorpus()
    sentences = vectorizer.transform(corpus)
    sentences = sentences.todense()
    sentences = np.asarray(sentences)
    if not os.path.exists('./qa'):
        os.makedirs('./qa')
    with open('./qa/sentences.npy', 'wb') as f:
        np.save(f, sentences)


def vectorize_input(input):
    with open('./tmp/tfidf.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    input = preprocessing.parse_txt(input)
    input = np.asarray(vectorizer.transform([input]).todense())
    return input

# test cosinus similarité
def similar_question(input):
    output = {}
    output['initial_question'] = input
    with open('./tmp/models_saved/reglog_l2.pkl', 'rb') as f:
        model = pickle.load(f)
    similar_question = []
    associated_answer = []
    raw_data = pd.read_csv('./data/SFR/messages_formated_QA.csv', sep=';')
    input = vectorize_input(input)
    target_name = preprocessing.get_classes_names('./data/SFR/messages_formated_cat.csv')
    intent = []
    proba = []
    for index, prob in sorted(enumerate(model.predict_proba(input)[0]), key=operator.itemgetter(1), reverse=True)[:3]:
        intent.append(target_name[int(index)])
        proba.append(prob)
    print 'intent detected à', proba, '%:', intent
    index = raw_data[raw_data['label'] == intent[0]].index.tolist()
    index.extend(raw_data[raw_data['label'] == intent[1]].index.tolist())
    index.extend(raw_data[raw_data['label'] == intent[2]].index.tolist())
    with open('./qa/sentences.npy', 'rb') as sent:
        sentences = np.array(np.load(sent))
    results = {}
    print '...comparing with', len(index), 'past questions'
    start_time = time.time()
    for k in index:
        results[k] = float(cosine_similarity(input.reshape(1,-1), sentences[k].reshape(1,-1)))
    run_time = time.time() - start_time
    print '...elapsed :', run_time
    for i, cosine_sim in sorted(results.items(), key=operator.itemgetter(1), reverse=True)[:10]:
        similar_question.append(raw_data['question'][i])
        associated_answer.append(raw_data['answer'][i])
    output['questions'] = []
    output['answers'] = []
    output['intent'] = intent
    output['accuracy'] = proba
    for k in range(9):
        output['questions'].append(similar_question[k])
        output['answers'].append(associated_answer[k])
    print json.dumps(output, indent=4)


if __name__ == '__main__':

    input = "Puis-je changer de carte sim"
    similar_question(input)
