import re
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from gensim import models
import numpy as np
import itertools


def text_to_word_list(text):
    text = str(text)
    text = text.lower()

    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.split()

    return text


def make_w2v_embeddings(df, embedding_dim=300, empty_w2v=False):
    vocabulary = {}
    vocabulary_counter = 0

    vocabulary_not_w2v = {}
    vocabs_not_w2v_counter = 0

    stops = set(stopwords.words('english'))

    print("Loading word2vec model(it may take britain out of the EU) ...")

    if empty_w2v:
        word2vec = EmptyWord2Vec
    else:
        word2vec = models.KeyedVectors.load_word2vec_format("./google/googleNews/GoogleNews-vectors-negative300.bin", binary=True)

    for index, row in df.iterrows():

        if index != 0 and index % 1000 == 0:
            print("{:,} sentences embedded.".format(index), flush=True)
        for question in ['lemmatized', 'query']:
        #for question in ['question1', 'question2']:
            q2n = []
            for word in text_to_word_list(row[question]):
                if word in stops:
                    continue

                if word not in word2vec.vocab:
                    if word not in vocabulary_not_w2v:
                        vocabs_not_w2v_counter += 1
                        vocabulary_not_w2v[word] = 1

                if word not in vocabulary:
                    vocabulary_counter += 1
                    vocabulary[word] = vocabulary_counter
                    q2n.append(vocabulary_counter)
                else:
                    q2n.append(vocabulary[word])

            df.at[index, question] = q2n

    embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)
    embeddings[0] = 0

    for word, index in vocabulary.items():
        if word in word2vec.vocab:
            embeddings[index] = word2vec.word_vec(word)
    del word2vec

    return df, embeddings


def split_and_zero_padding(df, max_seq_length):

    X = {'left': df['lemmatized'], 'right': df['query']}

    for dataset, side in itertools.product([X], ['left', 'right']):

        dataset[side] = pad_sequences(dataset[side], padding='pre', truncating='post', maxlen=max_seq_length)

    return dataset


class ManDist(Layer):
    def __init__(self, **kwargs):
        self.result = None
        super(ManDist, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ManDist, self).build(input_shape)

    def call(self, x, **kwargs):
        self.result = K.exp(-K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True))
        return self.result

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)


class EmptyWord2Vec:
    vocab = {}
    word_vec = {}
