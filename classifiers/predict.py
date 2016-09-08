# coding: utf8
# from classifiers.train import classifier
from preprocessing.parse import parse
import os, pickle, json, operator
import numpy as np
from pprint import pprint


def intent(sentence, model_name='reglog_l2?p=1.pkl', threshold=0.01):

    # we load tfidf (idf + vocabulary learn by fit in tfidf.py) & the model
    try:
        with open('./tmp/models_saved/'+model_name, 'rb') as f:
            model = pickle.load(f)
        with open('./tmp/tfidf.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        with open('./tmp/models_saved/classes.json', 'rb') as f:
            classes = json.load(f)
            target_names = [labels for key, labels in classes.items()]
    except:
        raise IOError('ML/DL model not found, try to train it from classifier/train.py')


    # we preprocess and vectorize the input
    text_vector = parse(sentence)
    text_vector = np.asarray(tfidf_vectorizer.transform([text_vector]).todense())

    # we apply the model on our vectorized input
    list_acc = model.predict_proba(text_vector)[0]
    accuracy = {target_names[index]: acc for index, acc in enumerate(list_acc)}
    sorted_acc = sorted(accuracy.items(), key=operator.itemgetter(1), reverse=True)
    # we use our threshold to determine if we understood the intent
    if sorted_acc[0][1] > threshold:
        comprehension = True
    else:
        comprehension = False

    results = {}
    results['intent'] = [label for label, acc in sorted_acc][0:4]
    results['accuracy'] = [acc for label, acc in sorted_acc][0:4]
    results['ok'] = comprehension
    # print pprint(results, indent=4)
    return results['intent'], results['accuracy'], results['ok']



if __name__ == '__main__':
    sentence = "J'ai envie d'aller faire une course"
    model_name = 'reglog_l1?p=1.0.pkl'
    intent(sentence)
