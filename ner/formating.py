# coding: utf8
from pprint import pprint
import re, pickle

# with open('./ner/data/ner_dataset_french_2.txt', 'r+') as f_2:
#     with open('./ner/data/ner_dataset_french.txt', 'r+') as f:
#         sentences_2 = f_2.readlines()
#         sentences = f.readlines()
#         new_sentences = sentences + list(set(sentences_2)-set(sentences))
#         print(len(sentences), len(sentences_2), len(new_sentences))
#         with open('./ner/data/ner_dataset_french_3.txt', 'w+') as output:
#             for line in new_sentences:
#                 output.write(line)


def group(lst, n):
  for i in range(0, len(lst), n):
    val = lst[i:i+n]
    if len(val) == n:
      yield tuple(val)


with open('./ner/data/ner_dataset_french_3.txt', 'r+') as f:
    sentences = f.readlines()
    # remove blank lines
    sentences = [sent for sent in sentences if sent != '\n']
    pprint(sentences[100:120])
    print "="*50
    for i in range(len(sentences)):
        # split by space and |
        sentences[i] = re.split('[ |]', sentences[i])
        # remove the carriage return \n
        sentences[i][-1] = sentences[i][-1].replace('\n', '')
    # transform I- in B- where entities Begins
    for i in range(len(sentences)):
        if sentences[i][2] != '0':
            sentences[i][2] = sentences[i][2].replace('I-', 'B-')
        for j in range(3, len(sentences[i])):
            if sentences[i][j] != 'O' and sentences[i][j-3] == 'O':
                sentences[i][j] = sentences[i][j].replace('I-', 'B-')
    # group by 3 (word, pos_tag, iob_tag)
    for i in range(len(sentences)):
        sentences[i] = list(group(sentences[i], 3))

    pprint(sentences[100:120])
    with open('./ner/data/ner_dataset_french_formated.pkl', 'wb') as f:
        pickle.dump(sentences, f)
