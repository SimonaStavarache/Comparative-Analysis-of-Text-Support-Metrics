import pandas as pd
import preprocessing
import embeddings
import tests
import ML_models
import distances


def main():

    """ Write in variable text your text """

    #text = "Stock market reached a historical low after the covid pandemic hit economies."
    #text = "Joe Biden is fighting Donald Trump's conspiracy theories. "
    #text = "The black lives matter protests show the extent of racial inequality."
    #text = "The covid vaccine search continues."
    text = "China territorial claims in the south china sea spark new conflicts."

    """ Loading the dataframe with tweets that is already preprocessed with embeddings"""

    tweet_data_frame = pd.read_pickle("./preprocessed_df")

    """ Loading database if no preprocessed_df exists"""

    # file = "./tweets.csv"
    # extended_tweet_data_frame = pd.read_csv(file, encoding='latin1', low_memory=False)
    # tweet_data_frame = extended_tweet_data_frame.iloc[:, 1:3].copy()
    # tweet_data_frame.columns = ["id", "tweet"]

    """ Preprocessing calls """

    # tweet_data_frame.drop_duplicates(subset=["tweet"], keep=False)
    # preprocessing.remove_unwanted_characters(tweet_data_frame)
    # print("Finished removing unwanted characters.....")
    # preprocessing.noise_removal(tweet_data_frame, "tweet")
    # print("Finished removing punctuation....")
    # preprocessing.lowercase(tweet_data_frame, "tweet")
    # print("Finished lowercasing....")
    # preprocessing.stopwords_removal(tweet_data_frame, "tweet", "stopwords_removed")
    # print("Soptwords removed ...")
    # preprocessing.lemmatization(tweet_data_frame, "stopwords_removed", "lemmatized_tokenized")
    # print("Finished lemmatizing data ...")
    # preprocessing.lemmatization_without_tokenization(tweet_data_frame, "stopwords_removed", "lemmatized")
    
    """ Embeddings calls """

    # embeddings.word2vec(tweet_data_frame, 0, "lemmatized_tokenized", "vectorized_cbow")
    # embeddings.word2vec(tweet_data_frame, 1, "lemmatized_tokenized", "vectorized_skip_gram")
    # embeddings.bert(tweet_data_frame, "lemmatized", "bert")
    # embeddings.glove(tweet_data_frame, "lemmatized", "glove")
    # print(tweet_data_frame)
    # tweet_data_frame = tweet_data_frame[tweet_data_frame["vectorized_skip_gram"].notna()]
    # tweet_data_frame = tweet_data_frame[tweet_data_frame["vectorized_cbow"].notna()]


    """ Save the new dataframe """
    # tweet_data_frame.to_pickle("./preprocessed_df")

    """ Train ML models """

    # ML_models.k_means(tweet_data_frame, "vectorized_cbow", "cbow")
    # ML_models.k_means(tweet_data_frame, "vectorized_skip_gram", "sg")
    # ML_models.lsi(tweet_data_frame, "lemmatized_tokenized", 2, 20, 1)
    # ML_models.lda_with_jensenn_distance(tweet_data_frame, "lemmatized_tokenized")
    # ML_models.siamese_manhattan_lstm_train()
    # ML_models.build_w2vec_model_Siamese(tweet_data_frame)
    # ML_models.wm_model(tweet_data_frame, "lemmatized_tokenized")
    # ML_models.train_svm()
    

    """ TESTS calls """
    # tests.start_jaccard(tweet_data_frame, text)
    # tests.start_cosine(tweet_data_frame, 0, 0, "vectorized_cbow", text)
    # tests.start_cosine(tweet_data_frame, 0, 1, "vectorized_skip_gram", text)
    # tests.start_cosine(tweet_data_frame, 1, 0, "bert", text)
    # tests.start_cosine(tweet_data_frame, 2, 0, "glove", text)
    # tests.k_means_test(tweet_data_frame, 0, text)
    # tests.k_means_test(tweet_data_frame, 1, text)
    # tests.siamese_lstm_test(tweet_data_frame, text)
    # tests.lsi_test(tweet_data_frame, "lemmatized_tokenized", text)
    # tests.word_mover_test(tweet_data_frame, 0, "lemmatized_tokenized", text)
    # tests.lda_shannon_test(tweet_data_frame, "topic_probability_distribution_tweet", text)
    # tests.combined_similarities_test(tweet_data_frame, "lemmatized_tokenized", text)



if __name__ == "__main__":
    main()
