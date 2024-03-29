# coding: utf-8
from preprocessing.parse import parse_soft
import pickle, re, json, os, sys, unicodedata
import numpy as np
from ner.predict import entities
from pprint import pprint
from classifiers.predict import predict
np.set_printoptions(threshold='nan')
np.set_printoptions(suppress=True)

script_dir = os.path.abspath(os.path.dirname(sys.argv[0]))
os.chdir(script_dir)

def isin(list_of_elem, sentence):
    for elem in list_of_elem:
        if elem in sentence:
            return True
    return False


def create_output(sentence):
    # get named entities (aka parameters)
    context_ner = entities(sentence)
    #sentence = parse_soft(sentence)
    sentence = parse_soft(sentence)

    # get intent
    greetingPattern = re.compile(r'((.)?onjour|b(.)?njour|bo(.)?jour|bon(.)?our|bonj(.)?ur|bonjo(.)?r|bonjou(.)?)|(.)?ello|(.)?alut')
    thanksPattern = re.compile(r'((.)?erci|mercis|thanks|thank you|thank)')
    if len(sentence) < 25 and re.search(greetingPattern, sentence) is not None:
        intents = ['greetings']
        intents, acc, comprehension = intents, [1], True
    elif len(sentence) < 25 and re.search(thanksPattern, sentence) is not None:
        intents = ['thanks']
        intents, acc, comprehension = intents, [1], True

    else:
        intents, acc, comprehension = predict(sentence, path='./tmp/intent', model_name='reglog_l2?p=5.0', deep=False, threshold=0.20)

    # detecting the tonality of the sentence
    if isin(['yes','oui','ouai',"d'acc"], sentence):
        tonalite = 'positive'
    elif isin(['no', 'na', 'hors'], sentence):
        tonalite = 'negative'
    else:
        tonalite = 'neutre'

    # create output to send to javascript
    context = {}
    context['message'] = sentence
    context['comprehension'] = comprehension
    context['accuracy'] = acc
    context['intent'] = intents
    context['entities'] = context_ner
    context['tonalite'] = tonalite
    print json.dumps(context)
    # pprint(context)
    # with open('output.json', 'wb') as f:
    #    json.dump(context, f)


if __name__ == '__main__':
    # sentence = "Je m'appelle Maxime Le Dantec et j'ai payé 15€ en plus lors de ma précédente facture...! comment c'est possible?'"
    sentence = sys.argv[1]
    create_output(sentence)
