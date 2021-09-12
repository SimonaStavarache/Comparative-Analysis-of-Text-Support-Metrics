from pyemd import emd
from numpy import dot
from numpy.linalg import norm
from scipy.spatial.distance import cdist
from scipy.spatial import distance
import numpy as np
from scipy.stats import entropy
from nltk.corpus import wordnet
import nltk
from nltk.corpus import wordnet_ic
from operator import itemgetter
import embeddings
from gensim.models import word2vec


nltk.data.path.append('./nltk_data/')

print("Brown wordnet is loading...")
brown_information_content = wordnet_ic.ic('ic-brown.dat')
print("Finished...")
print("Semcor wordnet is loading...")
semcor_information_content = wordnet_ic.ic('ic-semcor.dat')
print("Finished...")

# """ Needed for word mover's distance"""
# print("Google model loading...")
# google_model = word2vec.Word2VecKeyedVectors.load_word2vec_format("./google/gensim-data/word2vec-google-news-300/word2vec-google-news-300.gz", binary=True)
# google_model.init_sims(replace=True)
# print("Finished")


def jaccard_distance(sentence1, sentence2):

    sentence1 = sentence1.split(" ")
    sentence2 = sentence2.split(" ")
    intersection = set(sentence1).intersection(set(sentence2))
    union = set(sentence1).union(sentence2)

    jaccard = len(intersection)/len(union)

    return jaccard


def cosine_similarity(vector1, vector2):

    return dot(vector1, vector2)/(norm(vector1) * norm(vector2))


def manhattan_distance(vector1, vector2):

    return cdist(vector1, vector2, metric='cityblock')


def euclidean_distance(vector1, vector2):

    return distance.euclidean(vector1, vector2)


def jensen_shannon_distance(vector1, vector2):

    vector1 = np.array(vector1)
    vector2 = np.array(vector2)

    if len(vector1) > len(vector2):
        vector2.resize(vector1.shape)
    else:
        vector1.resize(vector2.shape)

    media = sum(vector1, vector2)/2

    divergence = (entropy(vector1, media) + entropy(vector2, media))/2

    return np.sqrt(divergence)


def word_movers_distance(sentence1, sentence2):

    dist = google_model.wmdistance(sentence1, sentence2)

    return dist


def max_similarity(word1, word2):

    word1 = wordnet.synsets(word1)
    word2 = wordnet.synsets(word2)

    w = wordnet.synsets("page")[0]

    similarity_normalisations = [w.lch_similarity(w), w.wup_similarity(w),
                                 w.lin_similarity(w, semcor_information_content), w.jcn_similarity(w, brown_information_content),
                                 w.jcn_similarity(w, semcor_information_content)]

    distance_synsets = []
    for i in word1:
        for j in word2:
            if i.pos() == j.pos():
                try:
                    lch_similarity = i.lch_similarity(j)/similarity_normalisations[0]
                    if lch_similarity is None:
                        lch_similarity = 0
                except:
                    lch_similarity = 0
                try:
                    wup_similarity = i.wup_similarity(j)/similarity_normalisations[1]
                    if wup_similarity is None:
                        wup_similarity = 0
                except:
                    wup_similarity = 0
                try:
                    lin_similarity = i.lin_similarity(j, semcor_information_content)/similarity_normalisations[2]
                    if lin_similarity is None:
                        lin_similarity = 0
                except:
                    lin_similarity = 0
                try:
                    jcn_similarity_1 = i.jcn_similarity(j, brown_information_content)/similarity_normalisations[3]
                    if jcn_similarity_1 is None:
                        jcn_similarity_1 = 0
                except:
                    jcn_similarity_1 = 0
                try:
                    jcn_similarity_2 = i.jcn_similarity(j, semcor_information_content)/similarity_normalisations[4]
                    if jcn_similarity_2 is None:
                        jcn_similarity_2 = 0
                except:
                    jcn_similarity_2 = 0

                distance_synsets.append([i, j, max(lch_similarity, lin_similarity, jcn_similarity_1, jcn_similarity_2, wup_similarity)])
            else:
                distance_synsets.append([i, j, 0])

    similarity_result = max(distance_synsets, key=itemgetter(2)) if distance_synsets else ["none", "none", 0]
    print(similarity_result)
    return similarity_result[2]


def combined_similarity(sentence1_list, sentence2_list, data_frame):
    """ Needs tokenization """

    first_sum = 0
    second_sum = 0
    first_idf_sum = 0
    second_idf_sum = 0

    term_frequency_dictionary = embeddings.term_frequency(data_frame)

    for word1 in sentence1_list:
        for word2 in sentence2_list:
            maximum_similarity = max_similarity(word1, word2)
            first_sum += maximum_similarity * term_frequency_dictionary[word1]
            first_idf_sum += term_frequency_dictionary[word1]

    for word2 in sentence2_list:
        for word1 in sentence1_list:
            maximum_similarity = max_similarity(word2, word1)
            second_sum += maximum_similarity * term_frequency_dictionary[word2]
            second_idf_sum += term_frequency_dictionary[word2]

    final_sum = (first_sum/first_idf_sum + second_sum/second_idf_sum)/2

    return final_sum
