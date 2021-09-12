from gensim.models import Word2Vec
from sent2vec.vectorizer import Vectorizer
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
import spacy
import numpy as np
from collections import Counter


def tf_idf(data_frame):

    # weighted average for word2vec to compute a sentence vector
    print(data_frame["lemmatized_tokenized"])
    dct = Dictionary(data_frame["lemmatized_tokenized"])
    corpus = [dct.doc2bow(line) for line in data_frame["lemmatized_tokenized"]]

    model = TfidfModel(corpus)
    corpus_tf = model[corpus]

    tfidf_dictionary = {dct.get(id): value for doc in corpus_tf for id, value in doc}

    return tfidf_dictionary


def term_frequency(data_frame):

    flat_list = [item for sublist in data_frame for item in sublist]
    frequency_dictionary = Counter(flat_list)

    return frequency_dictionary


def word2vec_sentence(tweet, tfidf_dictionary, model):

    avg_vector = []
    for word in tweet:
        try:
            vector = (tfidf_dictionary[word]) * model.wv[word]
        except KeyError:
            vector = 0
        avg_vector.append(vector)

    return np.average(avg_vector, axis=0)


def bert_sentence(tweet, tfidf_dictionary, vectorizer):

    avg_vector = []
    for word in tweet:
        try:
            vector = (tfidf_dictionary[word]) * vectorizer.bert(word)
        except KeyError:
            vector = 0
        avg_vector.append(vector)

    return np.average(avg_vector, axis=0)


def word2vec(data_frame, type_value, column, new_column):

    # needs tokenization
    tweets_list = data_frame[column].tolist()

    model = Word2Vec(tweets_list, size=100, window=10, min_count=1, sg=type_value)

    tfidf_dictionary = tf_idf(data_frame)

    data_frame[new_column] = data_frame[column].apply(lambda tweet: word2vec_sentence(tweet, tfidf_dictionary, model))

    print(data_frame[new_column])


def bert(data_frame, column, new_column):

    """no tokenization needed"""

    print("Bert embeddings started...")
    vectorizer = Vectorizer()
    # data_frame[new_column] = data_frame[column].apply(lambda tweet: vectorizer.bert(tweet))
    tfidf_dictionary = tf_idf(data_frame)

    data_frame[new_column] = data_frame[column].apply(lambda tweet: bert_sentence(tweet, tfidf_dictionary, vectorizer))

    print(data_frame[new_column])


def glove(data_frame, column, new_column):

    """no tokenization needed"""

    print("Glove embeddings started...")
    model = spacy.load('en_core_web_sm')
    data_frame[new_column] = data_frame[column].apply(lambda tweet: model(tweet).vector)
    print(data_frame[new_column])


def glove_sentence(text):

    model = spacy.load('en_core_web_sm')
    return model(text).vector

