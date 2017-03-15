# Generate random numbers and a test set

List files to move:
```
for i in $(curl -s "https://www.random.org/integer-sets/?sets=1&num=45&min=1&max=224&order=index&format=plain&rnd=new"); do ls | grep Control | cut -z -d"
" -f $i; echo -n " "; done
for i in $(curl -s "https://www.random.org/integer-sets/?sets=1&num=50&min=1&max=248&order=index&format=plain&rnd=new"); do ls | grep Bridge | cut -z -d"
" -f $i; echo -n " "; done
```

move those files with mv and fix symlinks int the test folder:
```
for i in *txt; do ln -sf ../../../../infos/Veroff/$i; done
```