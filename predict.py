# coding: utf-8
from preprocessing import parse_txt
import preprocessing
import pickle
import numpy as np
import re
import json
import os, sys
from ner.ner import get_parameters
# import cosine_similarity
np.set_printoptions(threshold='nan')
np.set_printoptions(suppress=True)

script_dir = os.path.abspath(os.path.dirname(sys.argv[0]))
os.chdir(script_dir)


def get_intent(sentence):
    target_name = ['Forfaits & Options',
                   'Evolution tarifaire',
                   "Changer d'offre",
                   'Appels/Au-delà/Hors Forfait',
                   'Régul/Remboursement/Geste Co',
                   'SFR Presse',
                   'Demande résil - Raisons personnelles',
                   'Achat Services SMS+ Internet+',
                   "Résiliation d'offre en cours",
                   'Infos Offres FIBRE']

    # we load tfidf (idf + vocabulary learn by fit in tfidf.py) & the model
    with open('./tmp/models_saved/reglog_l2.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('./tmp/tfidf.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    # we preprocess and vectorize the input
    text_vector = parse_txt(sentence)
    text_vector = np.asarray(vectorizer.transform([text_vector]).todense())
    # we apply the model on our vectorized input
    intent = target_name[int(model.predict(text_vector))]
    acc = model.predict_proba(text_vector)[0][int(model.predict(text_vector))]
    # we use our threshold to determine if we understood the client
    threshold = 0.25
    if acc > threshold:
        comprehension = True
    else:
        comprehension = False
    return intent, acc, comprehension

def create_output(sentence):
    # get named entities (aka parameters)
    context_ner = get_parameters(sentence)

    # get intent
    bonjourPattern = re.compile(r'((.)?onjour|b(.)?njour|bo(.)?jour|bon(.)?our|bonj(.)?ur|bonjo(.)?r|bonjou(.)?)|(.)?ello')
    if len(sentence) < 25 and re.search(bonjourPattern, sentence) is not None:
        intent = 'greetings'
        intent, acc, comprehension = intent, 1, True

    elif len(sentence) < 250 and (context_ner['phone'] != '' or context_ner['email'] != ''):
        intent = 'give_info'
        intent, acc, comprehension = intent, 1, True

    else:
        intent, acc, comprehension = get_intent(sentence)

    # create output to send to javascript
    context = {}
    context['ok'] = comprehension
    context['accuracy'] = acc
    context['intent'] = intent
    context['entities'] = context_ner
    print json.dumps(context, indent=4)
    # with open('output.json', 'wb') as f:
    #    json.dump(context, f)


if __name__ == '__main__':

    # sentence = "Je m'appelle Maxime Le Dantec et j'ai payé 15€ en plus lors de ma précédente facture...! comment c'est possible?'"
    sentence = sys.argv[1]
    create_output(sentence)
