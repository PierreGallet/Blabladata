# coding: utf-8
from preprocessing.parse import parse_soft
import pickle, re, json, os, sys, unicodedata
import numpy as np
from ner.predict import entities
from classifiers.predict import intent
np.set_printoptions(threshold='nan')
np.set_printoptions(suppress=True)

script_dir = os.path.abspath(os.path.dirname(sys.argv[0]))
os.chdir(script_dir)


def create_output(sentence):
    # get named entities (aka parameters)
    context_ner = entities(sentence)
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

    elif len(sentence) < 250 and (context_ner['phone'] != '' or context_ner['email'] != ''):
        intents = ['give_info']
        intents, acc, comprehension = intents, [1], True
    else:
        intents, acc, comprehension = intent(sentence)

    # create output to send to javascript
    context = {}
    context['message'] = sentence
    context['ok'] = comprehension
    context['accuracy'] = acc
    context['intent'] = intents
    context['entities'] = context_ner
    print json.dumps(context, indent=4)
    # with open('output.json', 'wb') as f:
    #    json.dump(context, f)


if __name__ == '__main__':

    # sentence = "Je m'appelle Maxime Le Dantec et j'ai payé 15€ en plus lors de ma précédente facture...! comment c'est possible?'"
    sentence = sys.argv[1]
    create_output(sentence)
