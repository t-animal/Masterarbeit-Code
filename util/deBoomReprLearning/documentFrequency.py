
import gensim
import numpy as np
import logging as log

def getDocumentFrequencies(corpus, keyedVectors):
    """Gets the document frequency in corpus of all words in keyedVectors. The corpus must have been preprocessed
       strlow-ered and words must be seperated by whitespace.

       :param corpus: the corpus to iterate over
       :type corpus: iterable of strings

       :param keyedVectors: the word embeddings model to user
       :type keyedVectors: gensim.keyedvectors.KeyedVectors

       :returns: an np.array of the size of the the words in keyed vectors
       """
    n_words = len(keyedVectors.vocab)
    freqs = np.zeros(n_words)

    for lineNo, line in enumerate(corpus):
        if lineNo % 10000 == 0:
            log.info("Processing line {}...".format(lineNo))

        words = set(line.split())
        for word in words:
            if word in keyedVectors:
                freqs[keyedVectors.vocab[word].index] += 1.0

    return freqs

def storeDocumentFrequencies(corpusFileName = "enwiki.txt", keyedVectorsFileName = "minimal", outputFileName = "docfreq.npy"):
    """Gets the document frequencies in corpus sotred in corpusFileName of all words in the keyedVectors
       stored in keyedVectorsFileName and stores them in outputFilename. The corpus must have been preprocessed
       strlow-ered and words must be seperated by whitespace.
       """

    model = gensim.models.Word2Vec.load_word2vec_format(filename, binary = keyedVectorsFile.endswith(".bin"))

    with open(corpusFileName, "r") as corpusFile, open(outputFileName, "w") as outputFile:
        np.save(outputFile, getDocumentFrequencies(corpus, keyedVectors))

def loadDocumentFrequencies(storageFileName = "docfreq.npy"):
    """Loads document frequencies stored by storeDocumentFrequencies"""

    with open(storageFileName) as storageFile:
        return np.load(storageFileName)


#aliases for easier importing
store = storeDocumentFrequencies
load = loadDocumentFrequencies


if __name__ == "__main__":
    from sys import argv

    storeDocumentFrequencies(argv[0], argv[1], argv[2])