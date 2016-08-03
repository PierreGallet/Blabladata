# Blabladata

## git stuff

You can go read this cheat sheet first: https://services.github.com/kit/downloads/fr/github-git-cheat-sheet.pdf

If you need to commit new change, process in this order:

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

* If you commited something that can't be push, and you are ahead of origin/master of n commit, use:
```git
git reset --soft HEAD~1
```
This will not erase your local directory (if you use --hard it will)
