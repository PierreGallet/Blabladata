# Blabladata

## dependencies

* crfsuite from https://github.com/tpeng/python-crfsuite. See an example here : http://nbviewer.jupyter.org/github/tpeng/python-crfsuite/blob/master/examples/CoNLL%202002.ipynb
* sklearn
* keras
* spacy
* pandas
* numpy


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

* git reset <file> provoque l'action inverse de git add <file>, il désindexe le file et permet de prendre celui du HEAD. Cf https://git-scm.com/book/fr/v2/Utilitaires-Git-Reset-d%C3%A9mystifi%C3%A9
