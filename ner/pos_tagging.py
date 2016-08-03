# coding: utf8
from pprint import pprint
from tree_tagger.treetagger_python2 import TreeTagger

def pos_tagging(sentence, language='french'):
    tagger = TreeTagger(language=language)
    return tagger.tag(sentence)

if __name__ == '__main__':
    print pos_tagging("Il aime les bananes et les pommes")
