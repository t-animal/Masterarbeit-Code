This only contains larger data sets/binaries which are not included in the git.

Currently, there's:

* GoogleNews-vectors-negative300.bin from http://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/
* enwiki-20161201-pages-articles.xml.bz2 from http://ftp.acc.umu.se/mirror/wikimedia.org/dumps/enwiki/20161201/

The first is one  too large to be saved via the save() method of the model (pickling throws a memory exception)!