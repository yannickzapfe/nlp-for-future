import os
import nltk
import numpy as np
import pandas as pd

from imblearn.under_sampling import RandomUnderSampler
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# Load first n book reviews including the review text and score in a dataframe.
def load_books_rating_data(first_n):
    if first_n and ('reviews' + str(first_n) + '.csv') in os.listdir('data'):
        return pd.read_csv('data/reviews' + str(first_n) + '.csv')

    books_data_raw = pd.read_csv('data/Books_rating.csv')

    books_data = books_data_raw[['review/score', 'review/text']].copy()
    books_data.rename(columns={'review/score': 'score', 'review/text': 'review'}, inplace=True)

    # Remove rows with null values.
    books_data = books_data.dropna()

    return books_data.head(first_n)


def tokenize(sents, remove_stop_words=True):
    sents_tokenized = [word_tokenize(sent.lower()) for sent in sents]

    if not remove_stop_words:
        return sents_tokenized

    # Remove english stop words.
    stop_words = list(stopwords.words("english")) + [",", ".", "(", ")", "'s", "&"]
    return [[word for word in sent if word not in stop_words] for sent in sents_tokenized]


# Pad tokenized review texts to have the same length.
def nn_pad(review_texts_tokenized, maxlen=None):
    return pad_sequences(review_texts_tokenized, maxlen=maxlen, padding="post")


def under_sample(features, scores):
    # Returns (features_balanced, scores_balanced).
    return RandomUnderSampler(random_state=42).fit_resample(features, scores)


# Flatten list of sentences to list of words.
def flatten(sents):
    return [word for sent in sents for word in sent]


def get_stop_words():
    return list(nltk.corpus.stopwords.words('english')) + [",", ".", "(", ")", "'s", "&"]


def remove_stop_words_from_words(words):
    return [word.lower() for word in words if word.lower() not in get_stop_words()]


def get_review_features(X_balanced, words, most_common_n=2000):
    words_freq_dist = nltk.FreqDist(words)
    words_map = {word for word, count in words_freq_dist.most_common(most_common_n)}
    return [{word: word in set(review) for word in words_map} for review in X_balanced]


def one_hot_encode(scores):
    return np_utils.to_categorical([int(score) - 1 for score in scores])


def create_embeddings_matrix(word_index):
    # Load embeddings from file.
    embeddings_dict = {}
    # TODO: Try 100d and pass as param to function.
    embeddings_file = open('data/glove.6B.300d.txt', encoding="utf8")
    for line in embeddings_file:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        embeddings_dict[word] = vector
    embeddings_file.close()

    # Adding 1 because of reserved 0 index
    vocab_size = len(word_index) + 1

    # Create embeddings matrix with words on one axis and vector values on the other.
    embeddings_matrix = np.zeros((vocab_size, 300))
    for word, index in word_index.items():
        embedding = embeddings_dict.get(word)
        if embedding is not None:
            embeddings_matrix[index] = embedding

    return embeddings_matrix


def nn_preprocess(first_n):
    books_data = load_books_rating_data(first_n)

    X_balanced, y_balanced = under_sample(np.array(books_data.text).reshape(-1, 1), books_data.score)
    X_balanced = [sent[0] for sent in X_balanced]

    X_tokenized = tokenize(X_balanced)

    vocab = set([word for sent in X_tokenized for word in sent])
    word_index = {}
    index = 1
    for word in vocab:
        word_index[word] = index
        index += 1

    X_indexed = [[word_index[word] for word in sent] for sent in X_tokenized]

    X_padded = nn_pad(X_indexed, None)

    embeddings_matrix = create_embeddings_matrix(word_index)

    return X_balanced, y_balanced, embeddings_matrix
