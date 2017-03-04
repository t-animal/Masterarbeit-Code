# Generate random numbers and a test set

List files to move:
```
for i in $(curl -s "https://www.random.org/integer-sets/?sets=1&num=20&min=1&max=99&order=index&format=plain&rnd=new"); do ls | grep -E pse[123] | cut -z -d"
" -f $i; echo -n " "; done
for i in $(curl -s "https://www.random.org/integer-sets/?sets=1&num=20&min=1&max=99&order=index&format=plain&rnd=new"); do ls | grep -E pse[456] | cut -z -d"
" -f $i; echo -n " "; done
```

move those files with mv and fix symlinks int the test folder:
```
for i in *txt; do ln -sf ../../../../infos/Veroff/$i; done
```