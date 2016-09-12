# coding: utf8
# from classifiers.train import classifier
""" predict intent & sentiment """
from preprocessing.parse import parse
import os, pickle, json, operator
import numpy as np
from pprint import pprint
from keras.models import model_from_json
from pprint import pprint
from keras.preprocessing import sequence

def predict(sentence, path='./tmp/sentiment', deep=True, model_name='cnn_lstm', threshold=0.01):
    if deep:
        # we load word2vec & the model
        try:
            with open(path+'/models_saved/'+model_name+'.json', 'rb') as f:
                model_json = f.read()
                print model_json
                model = model_from_json(model_json)
                model.load_weights(path+'/models_saved/'+model_name+'_weights.h5')
            with open(path+'/index_dict.pk') as f:
                index_dict = pickle.load(f)
            with open(path+'/models_saved/classes.json', 'rb') as f:
                classes = json.load(f)
                target_names = [labels for key, labels in classes.items()]
        except:
            raise IOError('DL model not found, try to train it from classifier/train.py')

        # we preprocess and vectorize the input
        text_parsed = parse(sentence)
        text_vector = []
        nb_unknown = 0
        print text_parsed
        for word in text_parsed.split():
            try:
                text_vector.append(index_dict[word.decode('utf-8')])
            except:
                text_vector.append(0)
                nb_unknown += 1
        # we apply the model on our vectorized input
        array = np.array([text_vector])
        array = sequence.pad_sequences(array, maxlen=150, truncating='post')
        list_acc = model.predict(array, batch_size=30, verbose=2)[0]

    else:
        # we load tfidf (idf + vocabulary learn by fit in tfidf.py) & the model
        try:
            with open(path+'/models_saved/'+model_name+'.pkl', 'rb') as f:
                model = pickle.load(f)
            with open(path+'/tfidf.pkl', 'rb') as f:
                tfidf_vectorizer = pickle.load(f)
            with open(path+'/models_saved/classes.json', 'rb') as f:
                classes = json.load(f)
                target_names = [labels for key, labels in classes.items()]
        except:
            raise IOError('ML model not found, try to train it from classifier/train.py')


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
    results['intent'] = [label for label, acc in sorted_acc][0:2]
    results['accuracy'] = [acc for label, acc in sorted_acc][0:2]
    results['ok'] = comprehension
    # print pprint(results, indent=4)
    return results['intent'], results['accuracy'], results['ok']

if __name__ == '__main__':

    sentence = "Je suis vraiment enervé, j'ai envie de partir de SFR. Votre service laisse vraiment à désirer"
    print 'sentiment: ', predict(sentence, path='./tmp/sentiment', model_name='cnn_lstm', deep=True)
    print 'intent: ', predict(sentence, path='./tmp/intent', model_name='reglog_l2?p=5.0', deep=False)
