# Generate random numbers and a test set

return int(filename[:-4]) not in ([9, 10, 11, 12] + list(range(85, 123)))

List files to move:
```
for i in $(curl -s "https://www.random.org/integer-sets/?sets=1&num=33&min=1&max=167&order=index&format=plain&rnd=new"); do ls | grep -vE "(009|01[012]|08[5-9]|09.|1[01].|12[0123])" | cut -z -d"
" -f $i; echo -n " "; done
for i in $(curl -s "https://www.random.org/integer-sets/?sets=1&num=33&min=1&max=168&order=index&format=plain&rnd=new"); do ls | grep -E "(009|01[012]|08[5-9]|09.|1[01].|12[0123])" | cut -z -d"
" -f $i; echo -n " "; done
```

move those files with mv and fix symlinks int the test folder:
```
for i in *txt; do ln -sf ../../../../infos/Veroff/$i; done
```
