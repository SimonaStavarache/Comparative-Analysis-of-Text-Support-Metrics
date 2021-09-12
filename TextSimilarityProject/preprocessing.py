import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WhitespaceTokenizer
from nltk.corpus import stopwords
from textblob import TextBlob
import manual_spellchecker
import nltk

nltk.data.path.append('./nltk_data/')


def lowercase(data_frame, column):

    data_frame[column] = data_frame[column].str.lower()


def noise_removal(data_frame, column):

    data_frame[column] = data_frame[column].replace(r"[$\|!@?\"',.:-]", ' ', regex=True)


def remove_unwanted_characters(data_frame):

    data_frame["tweet"] = data_frame["tweet"].str.replace("b'", "")\
                                             .str.replace("b\"", "")\
                                             .replace(r"\\n", ' ', regex=True)\
                                             .replace(r"(?<!\w)(http:\/\/)|(https:\/\/)[^\s]+", ' ', regex=True)
                                             #.replace(r"[\"',.:-]", ' ', regex=True)\
                                             #.str.lower()

    count = 0
    for index, row in data_frame.iterrows():
        a = re.sub(r'\\x[0-9]{0,2}[a-zA-Z]{0,2}[0-9]{0,2}', r'', row[1])
        data_frame.loc[index, "tweet"] = a
        print(data_frame.loc[index, "tweet"])
        count += 1

    print(data_frame)


def lemmatization(data_frame, column, new_column):

    wordnet_lemmatizer = WordNetLemmatizer()
    word_tokenizer = WhitespaceTokenizer()

    def text_lemma(tweet):
        return [wordnet_lemmatizer.lemmatize(word, pos="v") for word in word_tokenizer.tokenize(tweet)]

    data_frame[new_column] = data_frame[column].apply(text_lemma)

    print(data_frame)


def lemmatization_without_tokenization(data_frame, column, new_column):

    wordnet_lemmatizer = WordNetLemmatizer()
    word_tokenizer = WhitespaceTokenizer()

    def text_lemma(tweet):
        return str(' '.join([wordnet_lemmatizer.lemmatize(word, pos="v") for word in word_tokenizer.tokenize(tweet)]))

    data_frame[new_column] = data_frame[column].apply(text_lemma)

    print(data_frame)


def stopwords_removal(data_frame, column, new_column):

    lowercase(data_frame, column)
    stopwords_list = set(stopwords.words('english'))
    print(stopwords_list)
    data_frame[new_column] = data_frame[column].apply(lambda tweet: ' '.join([word for word in tweet.split() if word not in stopwords_list]))
    print(data_frame[new_column])


def manual_normalization(data_frame):

    obj = manual_spellchecker.spell_checker(data_frame, "tweet")
    print(obj.spell_check())
    data_frame["corrected"] = obj.correct_words()
    print(data_frame["corrected"])


def normalization(data_frame, column, new_column):

    data_frame[new_column] = data_frame[column].apply(lambda word: ''.join(TextBlob(word).correct()))
    print(data_frame[new_column])



