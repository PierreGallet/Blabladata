# coding: utf8
from tree_tagger.treetagger_python2 import TreeTagger


def tag(sentence, language='french'):
    tagger = TreeTagger(language=language)
    return tagger.tag(sentence)

def lemmatize(sentence, language='french'):
    tagger = TreeTagger(language=language)
    return ' '.join([lemma for word, pos_tag, lemma in tagger.tag(sentence)])

if __name__ == '__main__':
    print lemmatize("Il aime les bananes et les pommes.")
