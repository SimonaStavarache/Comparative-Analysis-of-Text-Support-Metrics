import tensorflow as tf
import time
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, Embedding, LSTM, GRU, Conv1D, Conv2D, GlobalMaxPool1D, Dense, Dropout
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from gensim.corpora import Dictionary
from gensim.models import LsiModel, Word2Vec
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.test.utils import get_tmpfile
from gensim.test.utils import common_texts
import pandas as pd
import random
from util import make_w2v_embeddings
from sklearn.model_selection import train_test_split
from util import split_and_zero_padding
from util import ManDist
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Bidirectional, Dense, Input, Lambda, LSTM, RepeatVector, TimeDistributed
from keras.models import Model
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
import preprocessing
from sklearn.metrics import classification_report
from keras import objectives, backend as K
import keras
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint
from keras.datasets import imdb
import os
from gensim.test.utils import common_texts
import tests


# def wm_model(data_frame, column_name):
#
#     #dictionary, term_matrix = prepare_corpus(data_frame[column_name], "dictionary_wm")
#
#     model = Word2Vec(common_texts, size=100, min_count=1)
#     model.train(data_frame[column_name], total_examples=15, epochs=2)
#     model.save("./Models/wm_model")

svm_model = pickle.load(open("./Models/svm_sentiment_model", 'rb'))
svm_vectorizer = pickle.load(open("./Models/svm_vectorizer", 'rb'))


def k_means(data_frame, column_name, file_name):

    final_list = []
    for item in data_frame[column_name]:
        aux_list = []
        for i in item:
            aux_list.append(i)
        final_list.append(aux_list)
    final_list = np.array(final_list)
    print(type(final_list), final_list)

    training_start_time = time.time()
    array = []

    for i in range(1, 5):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=32)
        kmeans.fit(final_list)
        array.append(kmeans.inertia_)


    training_end_time = time.time()
    print("Training time finished in %12.2f" % (training_end_time - training_start_time))

    plt.plot(range(1, 5), array)
    plt.title("Tuning")
    plt.xlabel("Number of clusters")
    plt.show()

    clusters = int(input("Choose the number of clusters: "))

    training_start_time = time.time()
    kmeans = KMeans(n_clusters=clusters, init='k-means++', random_state=32).fit(final_list)
    data_frame["cluster_" + file_name] = kmeans.predict(final_list)
    print(data_frame)

    pickle.dump(kmeans, open("./Models/k-means_model_" + file_name + ".sav", 'wb'))
    training_end_time = time.time()
    print("Training time finished in %12.2f" % (training_end_time - training_start_time))


def prepare_corpus(data_frame, dictionary_name):

    dictionary = Dictionary(data_frame)
    corpus = [dictionary.doc2bow(line) for line in data_frame]

    dictionary.save_as_text("./Models/" + dictionary_name)

    return dictionary, corpus


def plot_coherence(coherence_values, start, stop, step):

    x = range(start, stop, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Number of topics")
    plt.ylabel("Coherence values")
    plt.legend("coherence_values", loc='best')
    plt.show()


def lsi(data_frame, column_name, start, stop, step):

    """ Needs lemmatization with tokenization"""

    coherence_values = []
    model_list = []

    dictionary, term_matrix = prepare_corpus(data_frame[column_name], "dictionary_lsi")

    training_start_time = time.time()
    print("Starting coherence model....")

    for i in range(start, stop, step):

        model = LsiModel(term_matrix, num_topics=i, id2word=dictionary)
        model_list.append(model)
        coherence_model = CoherenceModel(model=model, texts=data_frame[column_name], dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherence_model.get_coherence())

    training_end_time = time.time()

    plot_coherence(coherence_values, start, stop, step)

    num_topics = input("Choose the number of clusters: ")

    print("Coherence model finished in %12.2f" % (training_end_time - training_start_time))

    training_start_time = time.time()
    lsi_model = LsiModel(term_matrix, num_topics=int(num_topics), id2word=dictionary)
    training_end_time = time.time()
    print("Training time finished in %12.2f" % (training_end_time - training_start_time))

    num_words = input("Choose how many words from a cluster: ")
    print(lsi_model.print_topics(num_topics=int(num_topics), num_words=int(num_words)))

    lsi_model.save("./Models/lsi.model")


def lda_with_jensenn_distance(data_frame, column_name):

    """ Needs lemmatization with tokenization"""

    # print("Heeeeere", type(data_frame[column_name]))
    #
    # new_dataframe = data_frame[column_name].to_list()
    # print(new_dataframe)
    # text1 = "Stock market reached a historical low after the covid pandemic hit economies"
    # text2 = "Joe Biden is fighting Donald Trump conspiracy theories "
    # text3 = "The black lives matter protests show the extent of racial inequality"
    # text4 = "The covid vaccine search continues"
    # text5 = "China territorial claims in the south china sea spark new conflicts"
    # add_dataframe = pd.DataFrame([text1, text2, text3, text4, text5], columns=['lemmatized_tokenized'])
    #
    # preprocessing.lowercase(add_dataframe, "lemmatized_tokenized")
    # preprocessing.stopwords_removal(add_dataframe, "lemmatized_tokenized", "lemmatized_tokenized")
    # preprocessing.lemmatization(add_dataframe, "lemmatized_tokenized", "lemmatized_tokenized")
    #
    # add_list = add_dataframe["lemmatized_tokenized"].to_list()
    # new_dataframe.append(add_list[0])
    # new_dataframe.append(add_list[1])
    # new_dataframe.append(add_list[2])
    # new_dataframe.append(add_list[3])
    # new_dataframe.append(add_list[4])
    #
    # #data_frame = pd.DataFrame(new_dataframe)
    #
    # x = pd.DataFrame()
    # x["lemmatized_tokenized"] = pd.Series(new_dataframe)
    # for index in x:
    #     x["lematized_tokenized"] = new_dataframe
    #
    # print("X heeeeere", x)

    corpus_dictionary, corpus_term_matrix = prepare_corpus(data_frame[column_name], "dictionary_lda") #x["lematized_tokenized"], "dictionary_lda")#

    training_start_time = time.time()
    lda_model = LdaModel(corpus_term_matrix, num_topics=20)

    training_end_time = time.time()
    print("Training time finished in %12.2f" % (training_end_time - training_start_time))

    lda_model.save("./Models/lda2.model")

    final_list = []
    for i in range(0, len(data_frame[column_name])): #x["lematized_tokenized"])):
        final_list.append(lda_model[corpus_term_matrix[i]])

    new_list = list_creation(final_list)

    for index, row in data_frame.iterrows():
        data_frame["topic_probability_distribution_tweet"] = new_list

    data_frame.to_pickle("./preprocessed_df")

    # tests.lda_shannon_test(data_frame, "topic_probability_distribution_tweet", lda_model, corpus_dictionary, text)


def list_creation(l):
    new_list = []
    for index1 in l:
        aux_list = [i[1] for i in index1]
        new_list.append(aux_list)
    print(new_list)
    return new_list


def build_w2vec_model_Siamese(data_frame):

    train_data, test_data = train_test_split(data_frame, test_size=0.2)
    print(train_data)
    print(test_data)


def siamese_manhattan_lstm_train():

    train_data = pd.read_csv("./MaLSTM_data/train.csv")
    for additional_column in ['question1', 'question2']:
        train_data[additional_column + '_n'] = train_data[additional_column]

    embedding_dim = 300
    max_seq_length = 20
    use_w2v = True

    train_data, embeddings = make_w2v_embeddings(train_data, embedding_dim=embedding_dim, empty_w2v=not use_w2v)

    validation_size = int(len(train_data) * 0.1)

    X = train_data[['question1_n', 'question2_n']]
    Y = train_data['is_duplicate']

    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)

    X_train = split_and_zero_padding(X_train, max_seq_length)
    X_validation = split_and_zero_padding(X_validation, max_seq_length)

    Y_train = Y_train.values
    Y_validation = Y_validation.values

    assert X_train['left'].shape == X_train['right'].shape
    assert len(X_train['left']) == len(Y_train)

    gpus = 1
    batch_size = 1024 * gpus
    n_epoch = 50
    n_hidden = 50

    x = Sequential()
    x.add(Embedding(len(embeddings), embedding_dim,
                    weights=[embeddings], input_shape=(max_seq_length,), trainable=False))
    x.add(LSTM(n_hidden))

    shared_model = x

    left_input = Input(shape=(max_seq_length,), dtype='int32')
    right_input = Input(shape=(max_seq_length,), dtype='int32')

    malstm_distance = ManDist()([shared_model(left_input), shared_model(right_input)])
    model = Model(inputs=[left_input, right_input], outputs=[malstm_distance])

    if gpus >= 2:
        model = tf.keras.utils.multi_gpu_model(model, gpus=gpus)

    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

    training_start_time = time.time()

    malstm_trained = model.fit([X_train['left'], X_train['right']], Y_train,
                               batch_size=batch_size, epochs=n_epoch,
                               validation_data=([X_validation['left'], X_validation['right']], Y_validation))
    training_end_time = time.time()
    print("Training time finished.\n%d epochs in %12.2f" % (n_epoch, training_end_time - training_start_time))

    model.save('./models/SiameseManhattanLSTM.h5')


def siamese_plots(malstm_trained):

    # not finished!!!
    plt.subplot(211)
    plt.plot(malstm_trained.history['acc'])
    plt.plot(malstm_trained.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot loss
    plt.subplot(212)
    plt.plot(malstm_trained.history['loss'])
    plt.plot(malstm_trained.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.tight_layout(h_pad=1.0)
    plt.savefig('./Plots/history-graph.png')

    print(str(malstm_trained.history['val_acc'][-1])[:6] +
          "(max: " + str(max(malstm_trained.history['val_acc']))[:6] + ")")
    print("Done.")


def train_svm():

    trainData = pd.read_csv("./SVM_datasets/train_sentiment.csv")
    testData = pd.read_csv("./SVM_datasets/test_sentiment.csv")

    vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)

    train_vectors = vectorizer.fit_transform(trainData['Content'])
    test_vectors = vectorizer.transform(testData['Content'])

    classifier_linear = svm.SVC(kernel='linear')
    t0 = time.time()
    classifier_linear.fit(train_vectors, trainData['Label'])
    t1 = time.time()
    prediction_linear = classifier_linear.predict(test_vectors)
    t2 = time.time()
    time_linear_train = t1 - t0
    time_linear_predict = t2 - t1

    pickle.dump(classifier_linear, open("./Models/svm_sentiment_model", 'wb'))
    pickle.dump(vectorizer, open("./Models/svm_vectorizer", 'wb'))

    print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))


def svm_sentiment_determination(input_string):

    review_vector = svm_vectorizer.transform([input_string])
    prediction = svm_model.predict(review_vector)
    print(input_string, str(prediction))
    return str(prediction)
