import spacy


nlp = spacy.load('en')



doc = nlp(u"Apples and oranges are similar. Boots and hippos aren't.")

def iter_products(docs):
    for doc in docs:
        for ent in doc.ents:
            if ent.label_ == 'PRODUCT':
                yield ent

def word_is_in_entity(word):
    return word.ent_type != 0

def count_parent_verb_by_person(docs):
    counts = defaultdict(defaultdict(int))
    for doc in docs:
        for ent in doc.ents:
            if ent.label_ == 'PERSON' and ent.root.head.pos == VERB:
                counts[ent.orth_][ent.root.head.lemma_] += 1
    return counts
