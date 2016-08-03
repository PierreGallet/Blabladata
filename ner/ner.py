# coding: utf8
from itertools import chain
import pickle, re, json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import pycrfsuite
from pprint import pprint
from sklearn.cross_validation import train_test_split, StratifiedKFold
from pos_tagging import pos_tagging
np.random.seed(1337)  # for reproducibility
np.set_printoptions(suppress=True)
np.set_printoptions(threshold='nan')
from preprocessing import parse_txt
import Levenshtein


# feature engeneering part
def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),  # to know whether the word only have uppercases
        'word.istitle=%s' % word.istitle(),  # to know whether the word start with an uppercase
        'word.isdigit=%s' % word.isdigit(),  # to know whether the word is a number
        'postag=' + postag,
        'postag[:2]=' + postag[:2],
    ]
    # adding infos about previous word
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:postag=' + postag1,
            '-1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('BOS')  # Beginning Of Sentence
    # adding infos about next word
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:postag=' + postag1,
            '+1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('EOS')  # End Of Sentence

    return features

def sent2features(sent):
    """
    take a list of tuple (word, pos_tag) and format it into the feature vector
    """
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]


# class to train the NER model (a CRF) on the train data.
class ner():

    def __init__(self, directory):
        self.directory = directory

    # format to features and split dataset
    def prepare_data(self):
        with open(self.directory + '/data/ner_dataset_french_formated.pkl', 'rb') as f:
            sentences = pickle.load(f)

        train_sents, test_sents = train_test_split(sentences, test_size=0.25, random_state=1000)

        pprint(train_sents[:3])
        pprint(test_sents[:3])

        print "nb of train sentences:", len(train_sents)
        print "nbof test sentences:", len(test_sents)

        self.X_train = [sent2features(s) for s in train_sents]
        self.y_train = [sent2labels(s) for s in train_sents]

        self.X_test = [sent2features(s) for s in test_sents]
        self.y_test = [sent2labels(s) for s in test_sents]

        pprint(self.X_train[:3])
        pprint(self.y_train[:3])


    # training part
    def train(self, c1=1.0, c2=1e-3, max_iterations=50, possible_transitions=True):
        # on instancie notre trainer
        trainer = pycrfsuite.Trainer(verbose=False)

        for xseq, yseq in zip(self.X_train, self.y_train):
            trainer.append(xseq, yseq)

        trainer.set_params({
            'c1': c1,   # coefficient for L1 penalty
            'c2': c2,  # coefficient for L2 penalty
            'max_iterations': max_iterations,  # stop earlier
            'feature.possible_transitions': possible_transitions  # include transitions that are possible, but not observed
        })

        print trainer.params()

        # on train sur les données injectées xseq yseq et on sauvegarde le model dans la racine.
        trainer.train(self.directory + '/ner.crfsuite')


    # evaluating part
    def predict(self):
        """
        Classification report for a list of BIO-encoded sequences.
        It computes token-level metrics and discards "O" labels.
        """
        # loading the tagger
        tagger = pycrfsuite.Tagger()
        tagger.open(self.directory + '/ner.crfsuite')

        # predicting on test data
        self.y_pred = [tagger.tag(xseq) for xseq in self.X_test]
        print self.y_test[:3]
        print self.y_pred[:3]

        # fit data, then transform multi class labels to binary labels.
        lb = LabelBinarizer()
        y_test_combined = lb.fit_transform(list(chain.from_iterable(self.y_test)))
        y_pred_combined = lb.transform(list(chain.from_iterable(self.y_pred)))
        print y_test_combined[:3]
        print y_pred_combined[:3]
        tagset = set(lb.classes_) - {'O'}  # we take off label O because it means no entities.
        tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])  # we sort the tagset
        class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

        print('\n\n## FINISHED ##')
        print('\nclassification report:')
        print classification_report(y_test_combined, y_pred_combined, labels=[class_indices[cls] for cls in tagset], target_names=tagset)


def predict_entities(sentences):

    # loading the tagger
    tagger = pycrfsuite.Tagger()
    tagger.open('./ner/ner.crfsuite')
    sentences_pos = pos_tagging(sentences)
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
            context[str(name_entities[i]) + ' at ' + str(i)] = entity.strip(' ')
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
    zipcodePattern = re.compile(r'(?P<zipcode>\s[0-9]{4,5}\s)')
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


def get_parameters(sentences):
    context_ner = predict_entities(sentences)
    context_regex = regex_detection(sentences)
    context = dict(context_ner.items() + context_regex.items())
    # print json.dumps(context, indent=4)
    return context


if __name__ == '__main__':

    # ner = ner(directory='./ner')
    # ner.prepare_data()
    # ner.train()
    # ner.predict()
    # predict_entities("Je vais en Espagne et à Madrid pour mes vacances")

    sentences = "Mon numéro c'est +332-34.5 43.232, Je me prénome Maxime Le Dantec. J'ai payé 15 euros 20; et je suis à la bnp Paribas et j'habite au 4 rue Biscornet à Paris 75012. Email : pierre.gallet@hotmail.fr. site : http://41mag.fr/regexp-php-les-8-expressions-regulieres-les-plus-utilisees.html. 12/12/2019"
    get_parameters(sentences)
