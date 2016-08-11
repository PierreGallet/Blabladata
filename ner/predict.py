# coding: utf8
import re, pycrfsuite
import sys, os
from preprocessing.pos_tagging import tag
from ner.train import sent2features, sent2tokens, ner
from pprint import pprint

def predict_entities(sentences):

    translate_entities = {}
    translate_entities['B-PER'] = 'person'
    translate_entities['B-ORG'] = 'organisation'
    translate_entities['B-MISC'] = 'miscellaneous'
    translate_entities['B-LOC'] = 'location'


    # loading the tagger
    tagger = pycrfsuite.Tagger()
    if not os.path.exists('./tmp/ner.crfsuite'):
        print('## Ner model not found ##\n>>> Be sure you have a "./data/NER/ner_dataset_french_3.txt" training dataset\n>>> Training ner...')
        ner = ner(directory='./data/NER/ner_dataset_french_3.txt')
        ner.prepare_data()
        ner.train()
        ner.predict()
        print('## ner training finished ##')

    tagger.open('./tmp/ner.crfsuite')
    sentences_pos = tag(sentences)
    sentences_feat = sent2features(sentences_pos)
    sentences_tok = sent2tokens(sentences_pos)
    name_entities = tagger.tag(sentences_feat)

    context = {}
    i = 0
    while i in range(len(name_entities)):
        if name_entities[i] != 'O':
            j = 0
            entity = ''
            while i + j < len(name_entities) and name_entities[i+j] != 'O':
                entity += sentences_tok[i+j] + ' '
                j += 1
            context[translate_entities[name_entities[i]]] = entity.strip(' ')
            i = i + j
        else:
            i += 1
    # print json.dumps(context, indent=4)
    return context

def regex_detection(sentences):
    phone = ''
    email = ''
    date = ''
    url = ''
    bank = ''
    money = ''
    zipcode = ''

    phonePattern = re.compile(r'(?P<phone>(0|\+33)[-.\s]*[1-9]([-.\s]*[0-9]){8})')  # \s = [ \t\n\r\f\v]
    emailPattern = re.compile(r'(?P<email>[A-Za-z0-9._-]+@[A-Za-z0-9._-]{2,}\.[a-z]{2,10})')   #TODO: ajout de àâäçéèêëîïôöûùüÿñæœ ?
    datePattern = re.compile(r'(?P<date>[0-3][0-9][-/.\s]([0-9]){2}[-/.\s]([0-9]){2,4})')  # JJ/MM/AAAA
    zipcodePattern = re.compile(r'(?P<zipcode>[0-9]{4,5})')
    urlPattern = re.compile(r'(?P<url>((http|https|ftp):\/\/)?([-\w]*\.)?([-\w]*)\.(aero|asia|biz|cat|com|coop|edu|gov|info|int|jobs|mil|mobi|museum|name|net|org|pro|tel|travel|arpa|[a-z]{2,3})\/[-_\w\/=?]*(\.[\w]{2,8})?)')
    moneyPattern = re.compile(r'(?P<money>\s\d{,4}([,.€]\d{,2})?(\s)?(€|euros|euro|e|cent|cents|centimes|centime)?(\s)?(\d{,2})?\s)')

    if re.search(phonePattern, sentences) is not None:
        phone = re.search(phonePattern, sentences).group('phone').replace('.', '').replace('-', '').replace(' ', '')
    if re.search(emailPattern, sentences) is not None:
        email = re.search(emailPattern, sentences).group('email')
    if re.search(datePattern, sentences) is not None:
        date = re.search(datePattern, sentences).group('date')
    if re.search(urlPattern, sentences) is not None:
        url = re.search(urlPattern, sentences).group('url')
    if re.search(zipcodePattern, sentences) is not None:
        zipcode = re.search(zipcodePattern, sentences).group('zipcode').strip()
    if re.search(moneyPattern, sentences) is not None:
        money = re.search(moneyPattern, sentences).group('money').strip()
        if '€' not in money and 'euro' not in money:
            money = ''

    with open('./ner/list_bank.txt', 'r+') as f:
        for line in f.readlines():
            if line.replace('\n', '').lower() in sentences.lower():
                bank = line.replace('\n', '')
                break

    context = {}
    context['phone'] = phone
    context['email'] = email
    context['date'] = date
    context['url'] = url
    context['bank'] = bank
    context['money'] = money
    context['zipcode'] = zipcode

    # print json.dumps(context, indent=4)
    return context


def entities(sentences):
    context_ner = predict_entities(sentences)
    context_regex = regex_detection(sentences)
    context = dict(context_ner.items() + context_regex.items())
    # print json.dumps(context, indent=4)
    return context


if __name__ == '__main__':

    sentences = "Mon numéro c'est +332-34.5 43.232, Je me prénome Maxime Le Dantec. J'ai payé 15 euros 20; et je suis à la bnp Paribas et j'habite au 4 rue Biscornet à Paris 75012. Email : pierre.gallet@hotmail.fr. site : http://41mag.fr/regexp-php-les-8-expressions-regulieres-les-plus-utilisees.html. 12/12/2019"
    pprint(entities(sentences))
