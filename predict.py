# coding: utf-8
from preprocessing import parse_txt
import preprocessing
import pickle
import numpy as np
import re
import json
np.set_printoptions(threshold='nan')
np.set_printoptions(suppress=True)

# we load tfidf (idf + vocabulary learn by fit in tfidf.py) & the model
with open('./tmp/models_saved/reglog_l2.pkl', 'rb') as f:
    model = pickle.load(f)
with open('./tmp/tfidf.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# we get labels names from the data that trained the model
data_directory = './data/SFR/messages_formated_cat.csv'
new_directory = './sfr'
preprocessing = preprocessing.prepocessing(data_directory, new_directory)
target_name = preprocessing.get_classes_names()

def parse_entitees(input):
    phone_number = ''
    email = ''

    phonePattern = re.compile(r'(?P<phone>[0-9. ]{9,15})')
    if re.search(phonePattern, input) is not None:
        phone_number = re.search(phonePattern, input).group('phone')

    emailPattern = re.compile(r'(?P<email>[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})')
    if re.search(emailPattern, input) is not None:
        email = re.search(emailPattern, input).group('email')
    return phone_number, email

def parse_intent(input):
    phone_number, email = parse_entitees(input)
    bonjourPattern = re.compile(r'((.)?onjour|b(.)?njour|bo(.)?jour|bon(.)?our|bonj(.)?ur|bonjo(.)?r|bonjou(.)?)')
    if len(input) < 25 and re.search(bonjourPattern, input) is not None:
        intent = 'greetings'
        return intent, 1, True
    elif len(input) < 50 and (phone_number or email):
        intent = 'give_info'
        return intent, 1, True
    else:
        # we preprocess and vectorize the input
        input = parse_txt(input)
        input = np.asarray(vectorizer.transform([input]).todense())
        # we apply the model on our vectorized input
        intent = target_name[int(model.predict(input))]
        proba = model.predict_proba(input)[0][int(model.predict(input))]
        # we use our threshold to determine if we understood the client
        threshold = 0.4
        if proba > threshold:
            comprehension = True
        else:
            comprehension = False
        return intent, proba, comprehension

def create_output(intent, proba, comprehension, phone_number, email):
    context = {}
    context['comprehension'] = comprehension
    context['intent'] = intent
    context['entitees'] = {}
    context['entitees']['phone_number'] = phone_number
    context['entitees']['email'] = email
    print json.dumps(context, indent=4)
    with open('output.json', 'wb') as f:
        json.dump(context, f)


if __name__ == '__main__':

    input = "Bonjour Ayaz et Julien"
    phone_number, email = parse_entitees(input)
    intent, proba, comprehension = parse_intent(input)
    print proba
    create_output(intent, proba, comprehension, phone_number, email)
