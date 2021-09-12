import distances
import pandas as pd
from util import make_w2v_embeddings
from util import split_and_zero_padding
from util import ManDist
import tensorflow as tf
from nltk.corpus import stopwords
import tables
import preprocessing
import embeddings
import pickle
import numpy as np
from gensim.models import LsiModel, Word2Vec, LdaModel
from gensim.corpora import Dictionary
from gensim import similarities
import ML_models
import time


def start_jaccard(data_frame, text):

    training_start_time = time.time()
    data_frame["query"] = text

    preprocessing.noise_removal(data_frame, "query")
    preprocessing.lowercase(data_frame, "query")
    preprocessing.lemmatization_without_tokenization(data_frame, "query", "query")

    data_frame["jaccard_distance"] = data_frame["tweet"].apply(
        lambda tweet: round(distances.jaccard_distance(tweet, data_frame["query"][0]), 2))

    new_data_frame = data_frame[["id", "tweet", "query", "jaccard_distance"]].copy()

    new_data_frame = new_data_frame.sort_values(by=["jaccard_distance"], ascending=False)

    columns = [2, 10, 7, 1.5]
    print(new_data_frame.head(15))
    tables.create_table(new_data_frame.head(15), "jaccard.png", columns)
    training_end_time = time.time()
    print("Prediction time finished in %12.2f" % (training_end_time - training_start_time))


def start_cosine(data_frame, embedding, word2vec_type, column, text):

    training_start_time = time.time()
    data_frame["query"] = text

    preprocessing.noise_removal(data_frame, "query")
    preprocessing.lowercase(data_frame, "query")
    preprocessing.stopwords_removal(data_frame, "query", "new_query")

    if embedding == 0:
        preprocessing.lemmatization(data_frame, "new_query", "new_query")
        embeddings.word2vec(data_frame, word2vec_type, "new_query", "new_query")
    elif embedding == 1:
        embeddings.bert(data_frame, "new_query", "new_query")
    elif embedding == 2:
        preprocessing.lemmatization_without_tokenization(data_frame, "new_query", "new_query")
        query_glove = embeddings.glove_sentence(data_frame["new_query"][0])

    data_frame["cosine_distance"] = data_frame[column].apply(
        lambda tweet: round(distances.cosine_similarity(tweet, data_frame["new_query"][0]), 2))

    new_data_frame = data_frame[["id", "tweet", "query", "cosine_distance"]].copy()
    print("New dataframe: ", new_data_frame["cosine_distance"])
    new_data_frame = new_data_frame.sort_values(by=["cosine_distance"], ascending=False)

    print(new_data_frame)
    columns = [2, 14.5, 5, 2]
    tables.create_table(new_data_frame.head(15), "cosine.png", columns)

    training_end_time = time.time()
    print("Prediction time finished in %12.2f" % (training_end_time - training_start_time))


def k_means_test(data_frame,  word2vec_type, text):

    data_frame["query"] = text

    preprocessing.noise_removal(data_frame, "query")
    preprocessing.lowercase(data_frame, "query")
    preprocessing.stopwords_removal(data_frame, "query", "new_query")
    preprocessing.lemmatization(data_frame, "new_query", "new_query")
    embeddings.word2vec(data_frame, word2vec_type, "new_query", "new_query")

    final_list = list()
    final_list.append(data_frame["new_query"][0])
    final_list = np.array(final_list)

    k_means_model = pickle.load(open("./Models/k-means_model_CBOW.sav", 'rb'))

    training_start_time = time.time()
    text_predicted = k_means_model.predict(final_list)
    training_end_time = time.time()
    print("Prediction time finished in %12.2f" % (training_end_time - training_start_time))

    print(text_predicted[0])

    clustered_dataframe = pd.read_pickle("./preprocessed_df")

    clustered_dataframe = clustered_dataframe[clustered_dataframe["cluster_cbow"] == text_predicted[0]]
    print(len(clustered_dataframe))
    print(clustered_dataframe.reset_index())


    """ Now we can use the new dataframe with different distances as optimization part"""

    sentiment_dataframe = sentiment_label(clustered_dataframe.reset_index(), text, "sentiment_label")
    # word_mover_test(clustered_dataframe.reset_index(), word2vec_type, "lemmatized_tokenized", text)
    word_mover_test(sentiment_dataframe, word2vec_type, "lemmatized_tokenized", text)
    print("hereeee", len(sentiment_dataframe))
    #combined_similarities_test(sentiment_dataframe, "lemmatized_tokenized", text)


def sentiment_label(data_frame, text, column_name):

    text_sentiment = ML_models.svm_sentiment_determination(text)

    # data_frame[column_name] = data_frame["lemmatized"].apply(lambda tweet: ML_models.svm_sentiment_determination(tweet))

    # print("Heeeeeereee data_frame", data_frame[column_name])
    #print("Heeeeeeereeeee text", text_sentiment)

    sentiment_labeled_dataframe = data_frame[data_frame[column_name] == text_sentiment]
    #print(sentiment_labeled_dataframe)

    return sentiment_labeled_dataframe.reset_index()


def lsi_test(data_frame, column, text):

    """ LSI with cosine similarity """

    training_start_time = time.time()
    data_frame["query"] = text
    load_model = LsiModel.load("./Models/lsi.model")
    # print(load_model)

    load_dictionary = Dictionary.load_from_text("./Models/dictionary_lsi")
    # print(load_dictionary)
    text = text.lower().split()

    load_corpus = [load_dictionary.doc2bow(line) for line in data_frame[column]]

    text_bow = load_dictionary.doc2bow(text)

    text_lsi = load_model[text_bow]
    print("Hereeeeee", load_model[load_corpus])
    index = similarities.MatrixSimilarity(load_model[load_corpus])
    text_sim = index[text_lsi]
    # print(text_sim)
    for i in range(0, len(index[text_lsi])):
        data_frame["lsi_similarity"] = index[text_lsi]

    new_data_frame = data_frame[["id", "tweet", "query", "lsi_similarity"]].copy()
    # print("New dataframe: ", new_data_frame["lsi_similarity"])
    new_data_frame = new_data_frame.sort_values(by=["lsi_similarity"], ascending=False)

    # print(new_data_frame)
    columns = [2, 10, 6, 2]
    tables.create_table(new_data_frame.head(15), "lsi.png", columns)

    training_end_time = time.time()
    print("Prediction time finished in %12.2f" % (training_end_time - training_start_time))


def word_mover_test(data_frame, word2vec_type, column, text):

    data_frame = sentiment_label(data_frame, text, "sentiment_label")

    training_start_time = time.time()
    print(text)

    data_frame["query"] = text

    preprocessing.noise_removal(data_frame, "query")
    preprocessing.lowercase(data_frame, "query")
    preprocessing.stopwords_removal(data_frame, "query", "new_query")
    preprocessing.lemmatization(data_frame, "new_query", "new_query")
    # embeddings.word2vec(data_frame, word2vec_type, "new_query", "new_query")
    print(data_frame["new_query"])
    text = data_frame["new_query"][0]
    print(text)

    data_frame["word_mover_distance"] = data_frame[column].apply(
        lambda tweet: round(distances.word_movers_distance(tweet, text), 2))

    new_data_frame = data_frame[["id", "tweet", "query", "word_mover_distance"]].copy()
    print("New dataframe: ", new_data_frame["word_mover_distance"])
    new_data_frame = new_data_frame.sort_values(by=["word_mover_distance"], ascending=True)

    print(new_data_frame)
    columns = [2, 10, 6, 2]
    tables.create_table(new_data_frame.head(15), "word_mover.png", columns)

    training_end_time = time.time()
    print("Prediction time finished in %12.2f" % (training_end_time - training_start_time))


def lda_shannon_test(data_frame, column_name, text):

    training_start_time = time.time()
    data_frame["query"] = text
    lda_model = LdaModel.load("./Models/lda2.model")
    corpus_dictionary = Dictionary.load_from_text("./Models/dictionary_lda")
    corpus = [corpus_dictionary.doc2bow(line) for line in data_frame["lemmatized_tokenized"]]

    preprocessing.noise_removal(data_frame, "query")
    preprocessing.lowercase(data_frame, "query")
    preprocessing.stopwords_removal(data_frame, "query", "new_query")
    preprocessing.lemmatization(data_frame, "new_query", "new_query")

    print(data_frame["new_query"][0])

    text_bow = corpus_dictionary.doc2bow(data_frame["new_query"][0])
    new_text_distribution = np.array([tup[1] for tup in lda_model.get_document_topics(bow=text_bow)])
    # prediction = [lda_model[text_bow]]

    # document_topic_distribution = np.array([[tup[1] for tup in lst] for lst in lda_model[corpus]])
    #
    # s = distances.jensen_shanon_matrix_dist(new_text_distribution, document_topic_distribution)
    # print(s.argsort()[:15])

    #text_list = ML_models.list_creation(new_text_distribution)

    data_frame["lda_shannon_distance"] = data_frame[column_name].apply(
        lambda tweet: round(distances.jensen_shannon_distance(tweet, new_text_distribution), 2))

    new_data_frame = data_frame[["id", "tweet", "query", "lda_shannon_distance"]].copy()
    print("New dataframe: ", new_data_frame["lda_shannon_distance"])
    new_data_frame = new_data_frame.sort_values(by=["lda_shannon_distance"], ascending=False)

    print(new_data_frame)
    columns = [2, 10, 6, 2]
    tables.create_table(new_data_frame.head(15), "lda_shannon.png", columns)

    training_end_time = time.time()
    print("Prediction time finished in %12.2f" % (training_end_time - training_start_time))


def siamese_lstm_test(data_frame, text):

    training_start_time = time.time()
    print("Start siamese ...")

    for tweet in data_frame["lemmatized"]:
        data_frame["test"] = data_frame["lemmatized"]
        data_frame["query"] = text
    print("Finished data preparation ...")

    embedding_dim = 300
    max_seq_length = 50
    data_frame, embeddings = make_w2v_embeddings(data_frame, embedding_dim=embedding_dim, empty_w2v=False)

    X_test = split_and_zero_padding(data_frame, max_seq_length)

    assert X_test['left'].shape == X_test['right'].shape

    model = tf.keras.models.load_model('./Models/SiameseManhattanLSTM.h5', custom_objects={'ManDist': ManDist})
    model.summary()

    data_frame["manhattan_distance"] = model.predict([X_test['left'], X_test['right']])

    print(data_frame)

    new_data_frame = data_frame[["id", "tweet", "query", "manhattan_distance"]].copy()
    print("New dataframe: ", new_data_frame["manhattan_distance"])
    new_data_frame = new_data_frame.sort_values(by=["manhattan_distance"], ascending=True)

    print(new_data_frame)
    columns = [2, 15, 4, 2]
    tables.create_table(new_data_frame.head(15), "MaLSTM.png", columns)

    training_end_time = time.time()
    print("Training time finished in %12.2f" % (training_end_time - training_start_time))


def combined_similarities_test(data_frame, column, text):

    data_frame = sentiment_label(data_frame, text, "sentiment_label")
    # data_frame = data_frame_old.iloc[0:400, :].copy()
    training_start_time = time.time()
    data_frame["query"] = text

    preprocessing.noise_removal(data_frame, "query")
    preprocessing.lowercase(data_frame, "query")
    preprocessing.stopwords_removal(data_frame, "query", "new_query")
    preprocessing.lemmatization(data_frame, "new_query", "new_query")

    text = data_frame["new_query"][0]

    data_frame["combined_similarity"] = data_frame[column].apply(
        lambda tweet: round(distances.combined_similarity(tweet, text, [tweet, text]), 2))

    new_data_frame = data_frame[["id", "tweet", "query", "combined_similarity"]].copy()
    print("New dataframe: ", new_data_frame["combined_similarity"])
    new_data_frame = new_data_frame.sort_values(by=["combined_similarity"], ascending=False)

    print(new_data_frame)
    columns = [2, 10, 7, 2]
    tables.create_table(new_data_frame.head(15), "combined_similarity.png", columns)

    training_end_time = time.time()
    print("Training time finished in %12.2f" % (training_end_time - training_start_time))



