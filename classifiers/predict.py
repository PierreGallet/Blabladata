# coding: utf8
# from classifiers.train import classifier
""" predict intent & sentiment """
from preprocessing.parse import parse
import os, pickle, json, operator
import numpy as np
from pprint import pprint

# Define a context manager to suppress stdout and stderr so that 'using theano backend' will not be printed (necessary for the use of python-shell that will look at everything that is printed)
class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])

with suppress_stdout_stderr():
    from keras.models import model_from_json
    from keras.preprocessing import sequence


def predict(sentence, path='./tmp/sentiment', deep=True, model_name='cnn_lstm', threshold=0.01):
    if deep:
        # we load word2vec & the model
        try:
            with open(path+'/models_saved/'+model_name+'.json', 'rb') as f:
                model_json = f.read()
                model = model_from_json(model_json)
                model.load_weights(path+'/models_saved/'+model_name+'_weights.h5')
            with open(path+'/index_dict.pk') as f:
                index_dict = pickle.load(f)
            with open(path+'/models_saved/classes.json', 'rb') as f:
                classes = json.load(f)
                target_names = [classes[str(i)] for i in range(len(list(classes.keys())))]
        except:
            raise IOError('DL model not found, try to train it from classifier/train.py')

        # we preprocess and vectorize the input
        text_parsed = parse(sentence)
        text_vector = []
        nb_unknown = 0
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
                target_names = [classes[str(i)] for i in range(len(list(classes.keys())))]
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
    sentence = "un chef d'oeuvre du cinema. Du grand spectacle, un plaisir pour les yeux!"
    # sentence = "Ce film a été vraiment nul , une merde du debut à la fin, mal filmé"
    print 'sentiment: ', predict(sentence, path='./tmp/sentiment', model_name='cnn_lstm', deep=True)
    print 'intent: ', predict(sentence, path='./tmp/intent', model_name='reglog_l2?p=5.0', deep=False)
