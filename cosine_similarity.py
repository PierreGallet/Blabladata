# coding: utf8
import pickle, operator, os, json, time
import numpy as np
import preprocessing
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


def vectorize_input(text):
    with open('./tmp/tfidf.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    text = preprocessing.parse_txt(text)
    text = np.asarray(vectorizer.transform([text]).todense())
    return text

# test cosinus similarité
def similar_question(text):
    """
    compute cosine similarity between the input text and our database of question from client, linked to operator answers.
    make a space of possible answers.
    based on the three best intent prediction to decrease the size of the question DB
    """
    # we will get the similar question and related answer there
    similar_question = []
    associated_answer = []
    similarity = []

    # we load the model
    with open('./tmp/models_saved/reglog_l2.pkl', 'rb') as f:
        model = pickle.load(f)

    # we load the question/answer database
    raw_data = pd.read_csv('./data/SFR/messages_formated_QA.csv', sep=';')

    # we vectorize the input
    text_vector = vectorize_input(text)
    target_name = preprocessing.get_classes_names('./data/SFR/messages_formated_cat.csv')

    # we get the three best predictions for the intent
    intent = []
    proba = []
    for index, acc in sorted(enumerate(model.predict_proba(text_vector)[0]), key=operator.itemgetter(1), reverse=True)[:3]:
        intent.append(target_name[int(index)])
        proba.append(acc)
    # print 'intent detected à', proba[0], '%:', intent[0]
    # print 'intent detected à', proba[1], '%:', intent[1]
    # print 'intent detected à', proba[2], '%:', intent[2]

    # and we only take the relevant data from question/answer database, thanks to the index variable
    index = raw_data[raw_data['label'] == intent[0]].index.tolist()
    index.extend(raw_data[raw_data['label'] == intent[1]].index.tolist())
    index.extend(raw_data[raw_data['label'] == intent[2]].index.tolist())

    # we also load the vectorized q/a database and we compute cosine similarity for every index.
    with open('./qa/sentences.npy', 'rb') as sent:
        sentences = np.array(np.load(sent))
    results = {}
    # print '...comparing with', len(index), 'past questions'
    start_time = time.time()
    for k in index:
        results[k] = float(cosine_similarity(text_vector.reshape(1,-1), sentences[k].reshape(1,-1)))
    run_time = time.time() - start_time
    # print '...elapsed :', run_time

    to_be_deleted = "Etes-vous toujours là ? N'ayant pas de réponse de votre part, je vais devoir fermer la conversation afin de répondre aux autres demandes. N'hésitez pas à nous recontacter par chat si vous avez besoin."
    # then we can append the 10 best results in ours lists, and then our json.
    for i, cosine_sim in sorted(results.items(), key=operator.itemgetter(1), reverse=True)[:10]:
        similar_question.append(raw_data['question'][i])
        associated_answer.append(raw_data['answer'][i].replace(to_be_deleted, ''))
        similarity.append(cosine_sim)
    output = {}
    output['initial_question'] = text
    output['questions'] = []
    output['answers'] = []
    output['intent'] = intent
    output['accuracy'] = proba
    output['similarity'] = []
    for k in range(9):
        output['questions'].append(similar_question[k])
        output['answers'].append(associated_answer[k])
        output['similarity'].append(similarity[k])
    # print json.dumps(output, indent=4)
    return associated_answer, similarity, similar_question

def similarity(text1, text2):
    text_vector_1 = vectorize_input(text1)
    text_vector_2 = vectorize_input(text2)
    print cosine_similarity(text_vector_1.reshape(1,-1), text_vector_2.reshape(1,-1))

if __name__ == '__main__':

    text = "je ne comprends pas"
    similarity(text, "je ne vois pas")
