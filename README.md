#TODO
Stefan nachfagen, wie das mit den Embeddings Trainieren war
Vortrag anfragen
Linear Discriminant Analysis
http://scikit-learn.org/stable/modules/lda_qda.html

# Lehrstuhl
Kisten mit viel Arbeitsspeiche:
* lme117.cs.fau.de
* lme118.cs.fau.de

# Implementierung
# Ideen:
* Gewichten mit abstand bei ngramsvm

## Word2Vec implementations
	- original version from google code is lost
	- C++ Implementierung https://github.com/jdeng/word2vec
	- Effiziente python implementierung (korrekt zitieren! siehe about) https://radimrehurek.com/gensim/about.html

## Gensim installieren
In ein virtualenv mit `pip install --upgrade gensim`


## Zitate:
* gensim: https://radimrehurek.com/gensim/about.html
* scikit-learn : http://scikit-learn.org/stable/about.html#citing-scikit-learn
* gnu-parallel

# Running tester in parallel:

parallel 'echo Windowsize: {1} Thresh: {2}; ./tester.py -c NGramSVM /storage/MA/GoogleNews-vectors-negative300.bin {1} {2} --load data/NGramSVM.GNews.{1}.Veroff.storage  --test Veroff --human 2>/dev/null | tee comprehensiveTest-{1}-{2}' ::: {40..1} ::: {3..5}.{0,5} ; cat comprehensiveTest-* > comprehensiveTest; rm comprehensiveTest-*


for i in {150..50}; do ./tester.py --human -c NGramSVM ~/ciptmp/GoogleNews-vectors-negative300.bin $i --train Veroff Winter --store ~/ciptmp/NGramSVM.GNews.$i.Veroff.Winter.storage; done

## NGramSVM validation runs:
Beste ergebnisse auf Veroff/Veroff (SVM trainiert in der Uni) mit:
Windowsize: 19 Thresh: 4.0
37 correct out of 54 ( 68.52%, p=0.00061)
Incorrect:                 17 ( 31.48%)
True positive aroused:     18 ( 66.67%)
True positive nonAroused:  19 ( 70.37%)
False positive aroused:     8 ( 29.63%)
False positive nonAroused:  9 ( 33.33%)
--
Windowsize: 17 Thresh: 4.5
37 correct out of 54 ( 68.52%, p=0.00061)
True positive aroused:     17 ( 62.96%)
True positive nonAroused:  20 ( 74.07%)
False positive aroused:     7 ( 25.93%)
False positive nonAroused: 10 ( 37.04%)
--
Windowsize: 17 Thresh: 5.0
37 correct out of 54 ( 68.52%, p=0.00061)
True positive aroused:     17 ( 62.96%)
True positive nonAroused:  20 ( 74.07%)
False positive aroused:     7 ( 25.93%)
False positive nonAroused: 10 ( 37.04%)
Windowsize: 17 Thresh: 5.5
37 correct out of 54 ( 68.52%, p=0.00061)
True positive aroused:     17 ( 62.96%)
True positive nonAroused:  20 ( 74.07%)
False positive aroused:     7 ( 25.93%)
False positive nonAroused: 10 ( 37.04%)
--
Windowsize: 16 Thresh: 4.5
37 correct out of 54 ( 68.52%, p=0.00061)
True positive aroused:     17 ( 62.96%)
True positive nonAroused:  20 ( 74.07%)
False positive aroused:     7 ( 25.93%)
False positive nonAroused: 10 ( 37.04%)
--
Windowsize: 16 Thresh: 5.0
37 correct out of 54 ( 68.52%, p=0.00061)
True positive aroused:     17 ( 62.96%)
True positive nonAroused:  20 ( 74.07%)
False positive aroused:     7 ( 25.93%)
False positive nonAroused: 10 ( 37.04%)