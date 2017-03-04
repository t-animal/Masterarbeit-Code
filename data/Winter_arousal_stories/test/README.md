# Generate random numbers and a test set

List files to move:
```
for i in $(curl -s "https://www.random.org/integer-sets/?sets=1&num=17&min=1&max=102&order=index&format=plain&rnd=new"); do ls | grep Winter_1 | cut -z -d"
" -f $i; echo -n " "; done
for i in $(curl -s "https://www.random.org/integer-sets/?sets=1&num=17&min=1&max=72&order=index&format=plain&rnd=new"); do ls | grep Winter_2 | cut -z -d"
" -f $i; echo -n " "; done
```

move those files with mv and fix symlinks int the test folder:
```
for i in *txt; do ln -sf ../../../../infos/Winter_arousal_stories/$i; done
```