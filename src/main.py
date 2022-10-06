#%%
# do imports
from collections import Counter
import pandas as pd
from io import StringIO
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split


# For a given review (in the form of a list or set of tokens), create a
# dictionary which tells us which words are present and which are not.
def get_review_features(review):
    review_words = set(review)
    return {word: word in review_words for word in words_map}


# flattens two dimensional list
def flatten(in_list):
    result = []
    for sent in in_list:
        [result.append(word) for word in sent]

    return result


#%%
books_data_raw = pd.read_csv('../data/Books_rating.csv')

#%%
len(books_data_raw)

#%%
books_data = books_data_raw[['review/score', 'review/text']].copy()
books_data.rename(columns={'review/score': 'score', 'review/text': 'text'}, inplace=True)

# TODO: use more 10000
books_data = books_data.head(20000)

#%%
books_data = books_data.dropna()
len(books_data)

#%%
# tokenize the review text
x = books_data.text.apply(lambda x: nltk.word_tokenize(x))

x.head()

#%%



#%%
words = flatten(x.tolist())

# remove all the stopwords
other_things_to_remove = [",", ".", "(", ")", "'s", "&"]
to_remove = list(stopwords.words('english')) + other_things_to_remove
words = filter(lambda i: i not in to_remove, [x.lower() for x in words])

#%%
# create a frequency distribution for the given words
words_freqDist = nltk.FreqDist(words)

# and put them into a map
words_map = {word for word, count in words_freqDist.most_common(2000)}


#%%
# create our pairs of features and target for every review
book_review_features = [(get_review_features(review), score) for review, score in zip(x, books_data.score)]


#%%
# split data into training and test
x_train, x_test = train_test_split(book_review_features, train_size=0.8)

#%%
# analyze the distribution of our data
c = Counter()
for item in x_train:
    c[item[1]] += 1

print(c)

#%%
# train a naive bayes on the training data and test its accuracy
b_naive_bayes = nltk.NaiveBayesClassifier.train(x_train)

acc = nltk.classify.accuracy(b_naive_bayes, x_test)

print(f"Accuracy: {round(acc * 100, 2)}%")
