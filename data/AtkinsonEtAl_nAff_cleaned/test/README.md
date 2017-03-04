# Generate random numbers and a test set

List files to move:
```
for i in $(curl -s "https://www.random.org/integer-sets/?sets=1&num=37&min=1&max=210&order=index&format=plain&rnd=new"); do ls | grep N | cut -z -d"
" -f $i; echo -n " "; done
for i in $(curl -s "https://www.random.org/integer-sets/?sets=1&num=37&min=1&max=156&order=index&format=plain&rnd=new"); do ls | grep R | cut -z -d"
" -f $i; echo -n " "; done
```

move those files with mv and fix symlinks int the test folder:
```
for i in *txt; do ln -sf ../../../../infos/AtkinsonEtAl_nAff_cleaned/$i; done
```