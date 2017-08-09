# Master's Thesis
This is the code repo of my Master's Thesis. I tried giving it a clear structure, but some
of it surely must be considered "academic code", sorry for that. Here's the structure in brief:

## Structure
* /classifiers: All classifier/feature vectors combinations talked about in the thesis reside here.
                All of them extend a base class from extend util/classifiers.py
* /data: stories must be stored here. I could not commit the stories I worked on to this repo, so 
         an interested reader must populate this folder
* /playground: In this folder I tried out new methods, classifiers, etc.
* /util: Two things are here: 1) some utility functions and classes, e.g. for creating cross validation
         datasets and distributing jobs across a cluster 2) some executables (see below)
         performing standalone utility functions like creating tables from JSON results

## Beginner's guide
1) Run ./install_dependencies.sh which should install a virtualenv with all necesarry dependencies
2) Set up a directory structure in ./data, replacing the current one. For licensing reasons (I guess), 
   the stories could not be committed to this repo, instead there are symlinks
3) Store the path to the word embeddings in tester.ini. As they amount to 50GB, they are not available
   online (cannot afford the possible traffic :( ) but can be requested. Or train them yourself
   using util/trainmodel.py
3) Play around with ./tester.py
4) Use ./nestedCrossVal.py and ./crossVal.py to replicate results

## Available tools
For ./tester and ./(nested)crossVal.py the following applies:
* For each dataset at least the function `isAroused` in util/__init__.py must be implemented. 
  Datasets will then be discovered automatically. 
* Classifiers are detected automatically from ./classifiers (at most once per file) and can
  be chosen by passing its classname to the -c flag
* The path to a word embedding must be put into ./tester.ini and then it can be selected by
  passing the chosen name to the -m flag
* In order to be used as a master or worker node, ./secretFile must be initialised with random data
  for authentication


This are the tools available:
* ./tester.py: Can be used to train, validate and crossvalidate a specific dataset. I used it mostly
               for testing modifications I made locally before cross validating it
* ./nestedCrossVal.py: Can be used to run a nested cross validation on a dataset. It supports
                       "local" mode, where all computation is done in parallel on the current machine.
                       "worker" mode, where it simply waits for a tuple of hyperparameters to cross
                       validate and
                       "master" mode, where it distributes all tuples of hyperparamters to the workers
                       The hyperparamters to optimize must be passed as JSON after --, e.g. 
                       `./nestedCrossVal.py -v local -c DocSumSVMClassifier -m your.model --datasets Veroff -- '{"SVM_C":[1,2,3]}'`
* ./crossVal.py: It's basically the same as nestedCrossVal.py but performs a simple cross validation

Furthermore some additional tools in ./util:

* ./util/calculateWinterBaseline.py: Quite uninteresting, produces the Winter baseline cross validation results
* ./util/storyscraper.py: Can be used to compile your own fanfiction corpus (cannot distribute my own, due to licensing)
* ./util/tablizeJSON.py: Can be used to turn JSON results from nested cross validation into pdfs
* ./util/trainmodel.py: Can be used to train GloVe and word2vec embeddings from wikipedia and fanfiction corpora
* ./util/weighPerAuthor.py: Can be used to convert JSON results per author instead per story
