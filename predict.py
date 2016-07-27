# coding: utf-8
from preprocessing import parse_txt
import preprocessing
import pickle
import numpy as np
import re
import json
import os, sys
np.set_printoptions(threshold='nan')
np.set_printoptions(suppress=True)

script_dir = os.path.abspath(os.path.dirname(sys.argv[0]))

threshold = 2

# we load tfidf (idf + vocabulary learn by fit in tfidf.py) & the model
with open(script_dir + '/tmp/models_saved/reglog_l2.pkl', 'rb') as f:
    model = pickle.load(f)
with open(script_dir + '/tmp/tfidf.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# we get labels names from the data that trained the model
#data_directory = './data/SFR/messages_formated_cat.csv'
#new_directory = './sfr'
#preprocessing = preprocessing.prepocessing(data_directory, new_directory)
#target_name = preprocessing.get_classes_names()

def parse_entitees(text):
    phone_number = ''
    email = ''

    phonePattern = re.compile(r'(?P<phone>[0-9. ]{9,15})')
    if re.search(phonePattern, text) is not None:
        phone_number = re.search(phonePattern, text).group('phone')

    emailPattern = re.compile(r'(?P<email>[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})')
    if re.search(emailPattern, text) is not None:
        email = re.search(emailPattern, text).group('email')
    return phone_number, email

def parse_intent(text):
    phone_number, email = parse_entitees(text)
    bonjourPattern = re.compile(r'((.)?onjour|b(.)?njour|bo(.)?jour|bon(.)?our|bonj(.)?ur|bonjo(.)?r|bonjou(.)?)')
    if len(text) < 25 and re.search(bonjourPattern, text) is not None:
        intent = 'greetings'
        return intent, 1, True
    elif len(text) < 50 and (phone_number or email):
        intent = 'give_info'
        return intent, 1, True
    else:
        # we preprocess and vectorize the input
        text = parse_txt(text)
        text = np.asarray(vectorizer.transform([text]).todense())
        # we apply the model on our vectorized input
        intent = "ELSE" #target_name[int(model.predict(text))]
        proba = model.predict_proba(text)[0][int(model.predict(text))]
        # we use our threshold to determine if we understood the client
        if proba > threshold:
            comprehension = True
        else:
            comprehension = False
        return intent, proba, comprehension

def create_output(intent, proba, comprehension, phone_number, email):
    context = {}
    context['ok'] = comprehension
    context['intent'] = intent
    context['entities'] = {}
    context['entities']['phone_number'] = phone_number
    context['entities']['email'] = email
    print json.dumps(context)
    #with open('output.json', 'wb') as f:
    #    json.dump(context, f)


if __name__ == '__main__':

    #text = str(raw_input("Coucou"))
    text = sys.argv[1]
    phone_number, email = parse_entitees(text)
    intent, proba, comprehension = parse_intent(text)
    create_output(intent, proba, comprehension, phone_number, email)
