# Blabladata

## project folder organization

The project is organized through 9 folders:
* ```/classifiers```, that groups the different machine learning & deep learning algorithms, using Keras & Sk-learn. There you can train, using data in a ```/data``` folder (that you have to create) and predict using a model from ```/tmp```.
* ```/embeddings```, that groups TF-IDF, word2vec, doc2vec and others way to embed ours word vectors, using Gensim & NLTK
* ```/formatting```, that is there to format any data to a csv with 2 columns (label, sentence) that is our format to do supervised learning.
* ```/preprocessing```, that process the raw words to the words that will be used for the embeddings part (part of dictionary, etc...)
* ```/ner```, that groups all ner algorithms, using CRF Suite for python
* ```/unsupervised```, that groups all unsupervised learning technics, for now : cosine similarity & kmeans.
* ```/paraphrase_detection```, that groups paraphrase detection algorithms.
* ```/misc```, that groups miscellaneous scripts
* ```/tmp```, that groups all ours templates (models that are trained).

## dependencies

* crfsuite from https://github.com/tpeng/python-crfsuite. See an example here : http://nbviewer.jupyter.org/github/tpeng/python-crfsuite/blob/master/examples/CoNLL%202002.ipynb
* sklearn
* keras
* spacy
* pandas
* numpy
* matplotlib
* gensim
* nltk
* Levenshtein from https://pypi.python.org/pypi/python-Levenshtein. Do ```python setup.py build``` to create the executable binary, then ```python setup.py install``` to copy them in site_package (repo used by python for packages)

les imports sont tous absolus: ils partent de ./Blabladata.

## git stuff

You can go read this cheat sheet first: https://services.github.com/kit/downloads/fr/github-git-cheat-sheet.pdf

If you need to commit new change, don't forget to put your heavy data file in the .gitignore and then process in this order:

* First : retrieve what is on the git that you don't have. Stash command will merge for you what can be easily merge (new files) and will give you 2 differents code in your files for things that are changed on the git and on your side
```git
git stash
git pull
git stash pop
```

* Second : you can commit and push, after merging manually what need to be merge manually
```git
git add .
git commit -m 'message to be commited'
git push
```

* If you commited something that can't be push (file > 50MO for instance), and you are ahead of origin/master of n commit, use:
```git
git reset --soft HEAD~1
```
This will not erase your local directory (if you use --hard it will)
