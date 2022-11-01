# tokenize the review text
from collections import Counter
from operator import itemgetter

import nltk
from imblearn.under_sampling import RandomUnderSampler
from nltk.corpus import stopwords
from sklearn.metrics import recall_score, accuracy_score, precision_score
from sklearn.model_selection import train_test_split

import preprocessing
from src.helpers import read_base_data, flatten, print_task_header

books_data = read_base_data(f'../data/Reviews_data/reviews{1000}.csv')
x = books_data.review.apply(lambda x: nltk.word_tokenize(x))
scores = [1.0, 2.0, 3.0, 4.0, 5.0]


def get_review_features(review, most_common_review_words):
    review_words = set(review)

    count_most_common = sum({word: word in review_words for word in words_map}.values())

    features = {
        'length': len(review),
        'most_common': count_most_common
    }
    for index, score in enumerate(scores):
        key = f'most_common_score_{str(score)}'
        current = most_common_review_words[index]
        value = sum({word: word in review_words for word in current}.values())
        features[key] = value / len(current)

    # features = {word: word in review_words for word in words_map}
    # features['len'] = len(review)
    return features


words = flatten(x.tolist())

# remove all the stopwords
other_things_to_remove = [",", ".", "(", ")", "'s", "&"]
to_remove = list(stopwords.words('english')) + other_things_to_remove
words = filter(lambda i: i not in to_remove, [x.lower() for x in words])

# %%
# create a frequency distribution for the given words
words_freqDist = nltk.FreqDist(words)

# and put them into a map
words_map = {word for word, count in words_freqDist.most_common(2000)}

most_common_review_words = []
for score in scores:
    x_score = flatten(preprocessing.tokenize(books_data[books_data.score == score].review))
    freq_dist = nltk.FreqDist(x_score)
    most_common_review_words.append({word for word, _ in freq_dist.most_common()})

u = set.intersection(*most_common_review_words)

for index, c in enumerate(most_common_review_words):
    tmp = most_common_review_words[:index] + most_common_review_words[index + 1:]
    most_common_review_words[index] = c - set.union(*tmp)


# %%
# create our pairs of features and target for every review
book_review_features = [(get_review_features(review, most_common_review_words), score) for review, score in zip(x, books_data.score)]

# %%
# analyze the distribution of our data
c = Counter()
for item in book_review_features:
    c[item[1]] += 1

min_key, min_count = min(c.items(), key=itemgetter(1))

# %%
# split data into training and test
x_train, x_test = train_test_split(book_review_features, train_size=0.8)

# %%
# train a naive bayes on the training data and test its accuracy
b_naive_bayes = nltk.NaiveBayesClassifier.train(x_train)

# temporarily remove labels from x_test
x_test_data = [x[0] for x in x_test]
x_test_scores = [x[1] for x in x_test]

pred = b_naive_bayes.classify_many(x_test_data)
acc = accuracy_score(x_test_scores, pred)
prec = precision_score(x_test_scores, pred, average='micro', zero_division=0)
rec = recall_score(x_test_scores, pred, average='micro', zero_division=0)
# print(f"predict: {pred}")
# print(f"actuals: {x_test_scores}")

print_task_header("Unbalanced dataset")
print(f"class count before balancing:\n{c}")

print(f"accuracy: {round(acc * 100, 2)}%")
print(f"precision: {round(prec * 100, 2)}%")
print(f"recall: {round(rec * 100, 2)}%")

X_raw = books_data.review
X_list = [[x] for x in X_raw]
y_raw = books_data.score
# %%

under_sampler = RandomUnderSampler(random_state=42)
X_balanced, y_balanced = under_sampler.fit_resample(X_list, y_raw)

book_review_features_b = [(get_review_features(review, most_common_review_words), score) for review, score in zip(X_balanced, y_balanced)]

c = Counter()
for item in book_review_features_b:
    c[item[1]] += 1

x_train_b, x_test_b = train_test_split(book_review_features, train_size=0.8)

# %%
# train a naive bayes on the training data and test its accuracy
b_naive_bayes_b = nltk.NaiveBayesClassifier.train(x_train_b)

# temporarily remove labels from x_test
x_test_data_b = [x[0] for x in x_test_b]
x_test_scores_b = [x[1] for x in x_test_b]

pred_b = b_naive_bayes.classify_many(x_test_data_b)
acc_b = accuracy_score(x_test_scores_b, pred_b)
prec_b = precision_score(x_test_scores_b, pred_b, average='micro', zero_division=0)
rec_b = recall_score(x_test_scores_b, pred_b, average='micro', zero_division=0)

print_task_header("Balanced dataset")
print(f"class count after balancing:\n{c}")
print(f"accuracy: {round(acc_b * 100, 2)}%")
print(f"precision: {round(prec_b * 100, 2)}%")
print(f"recall: {round(rec_b * 100, 2)}%")



        # load_tokenizer = AutoTokenizer.from_pretrained("./results/checkpoint-3000")

        # books_review_list = books_data.review.tolist()[:50]
        # books_score_list = books_data.score.tolist()[:50]
        # print_task_header(f"testing with {length} datapoints:")
        #
        # classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        # if you have GPU access, you can set the device parameter to 0 to use the GPU,
        # which will speed up model performance.
        # t_start = time.time()
        # pipe = TextClassificationPipeline(model=load_model, tokenizer=tokenizer)
        # # classifier = pipeline("zero-shot-classification", model=load_model, device=0, tokenizer=tokenizer)
        # t_end = time.time()
        # print(f"initializing classifier took {round(t_end - t_start, 2)}s")

        # labels = preprocessing.one_hot_encode([1.0, 2.0, 3.0, 4.0, 5.0])

        # for result in results:
        #     print(result)
        #     predictions.append(float(result['labels'][0]))

        # print(predictions)
        # print(books_score_list)
        # # tokenize the review text
        # x = preprocessing.tokenize(books_data.review)
        #
        # most_common_review_words = []
        # for score in scores:
        #     x_score = flatten(preprocessing.tokenize(books_data[books_data.score == score].review))
        #     freq_dist = nltk.FreqDist(x_score)
        #     most_common_review_words.append({word for word, _ in freq_dist.most_common()})
        #
        # u = set.intersection(*most_common_review_words)
        #
        # for index, c in enumerate(most_common_review_words):
        #     tmp = most_common_review_words[:index] + most_common_review_words[index + 1:]
        #     most_common_review_words[index] = c - set.union(*tmp)
        #
        # # %%
        # words = flatten(x)
        #
        # # %%
        # # create a frequency distribution for the given words
        # words_freqDist = nltk.FreqDist(words)
        #
        # # and put them into a map
        # words_map = {word for word, _ in words_freqDist.most_common(2000)}
        #
        # book_review_features = [(get_review_features(review, most_common_review_words)) for review, score
        #                         in zip(books_data.review, books_data.score)]
        #
        # df_features = pd.DataFrame(book_review_features)
        #
        # df_features_standardized = pd.DataFrame(StandardScaler().fit_transform(df_features), columns=df_features.columns)
        #
        # features_dict = df_features_standardized.to_dict(orient='records')
        #
        # train_features = [(feature, score) for feature, score
        #                   in zip(features_dict, books_data.score)]
        #
        # x_train_b, x_test_b = train_test_split(train_features, train_size=0.8)
        #
        # # %%
        # # train a naive bayes on the training data and test its accuracy
        # b_naive_bayes_b = nltk.NaiveBayesClassifier.train(x_train_b)
        #
        # # temporarily remove labels from x_test
        # x_test_data_b = [x[0] for x in x_test_b]
        # x_test_scores_b = [x[1] for x in x_test_b]
        #
        # pred_b = b_naive_bayes_b.classify_many(x_test_data_b)

# %%

# lengths = {
#     1.0: [],
#     2.0: [],
#     3.0: [],
#     4.0: [],
#     5.0: [],
# }
# for index, row in books_data.iterrows():
#     lengths[row['score']].append(len(row['text']))
#
# print(f"average lengths:")
# for k, v in lengths.items():
#     print(f"{k}: {np.array(v).mean()}")


# for index, row in books_data[books_data['score'] == 1.0].iterrows():
#     print(row['text'])


# for i in range(5):
#     with open(f'../data/reviews_{i + 1}.txt', 'w') as text_file_1:
#         for index, row in books_data[books_data['score'] == float(i+1)].iterrows():
#             text_file_1.write(row['text'])
#             text_file_1.write("\n\n")


# %%
# tokenize the review text
# x = books_data.review.apply(lambda x: nltk.word_tokenize(x))
#
# # x.head()
#
# # %%
#
#
# # %%
# words = flatten(x.tolist())
#
# # remove all the stopwords
# other_things_to_remove = [",", ".", "(", ")", "'s", "&"]
# to_remove = list(stopwords.words('english')) + other_things_to_remove
# words = filter(lambda i: i not in to_remove, [x.lower() for x in words])
#
# # %%
# # create a frequency distribution for the given words
# words_freqDist = nltk.FreqDist(words)
#
# # and put them into a map
# words_map = {word for word, count in words_freqDist.most_common(2000)}
#
# # %%
# # create our pairs of features and target for every review
# book_review_features = [(get_review_features(review), score) for review, score in zip(x, books_data.score)]
#
# # %%
# # analyze the distribution of our data
# c = Counter()
# for item in book_review_features:
#     c[item[1]] += 1
#
# min_key, min_count = min(c.items(), key=itemgetter(1))
#
# # %%
# # split data into training and test
# x_train, x_test = train_test_split(book_review_features, train_size=0.8)
#
# # %%
# # train a naive bayes on the training data and test its accuracy
# b_naive_bayes = nltk.NaiveBayesClassifier.train(x_train)
#
# # temporarily remove labels from x_test
# x_test_data = [x[0] for x in x_test]
# x_test_scores = [x[1] for x in x_test]
#
# pred = b_naive_bayes.classify_many(x_test_data)
# acc = accuracy_score(x_test_scores, pred)
# prec = precision_score(x_test_scores, pred, average='micro', zero_division=0)
# rec = recall_score(x_test_scores, pred, average='micro', zero_division=0)
# # print(f"predict: {pred}")
# # print(f"actuals: {x_test_scores}")
#
# print_task_header("Unbalanced dataset")
# print(f"class count before balancing:\n{c}")
#
# print(f"accuracy: {round(acc * 100, 2)}%")
# print(f"precision: {round(prec * 100, 2)}%")
# print(f"recall: {round(rec * 100, 2)}%")
#
# X_raw = books_data.text
# X_list = [[x] for x in X_raw]
# y_raw = books_data.score
# # %%
#
# under_sampler = RandomUnderSampler(random_state=42)
# X_balanced, y_balanced = under_sampler.fit_resample(X_list, y_raw)
#
# book_review_features_b = [(get_review_features(review), score) for review, score in zip(X_balanced, y_balanced)]
#
# c = Counter()
# for item in book_review_features_b:
#     c[item[1]] += 1
#
# x_train_b, x_test_b = train_test_split(book_review_features, train_size=0.8)
#
# # %%
# # train a naive bayes on the training data and test its accuracy
# b_naive_bayes_b = nltk.NaiveBayesClassifier.train(x_train_b)
#
# # temporarily remove labels from x_test
# x_test_data_b = [x[0] for x in x_test_b]
# x_test_scores_b = [x[1] for x in x_test_b]
#
# pred_b = b_naive_bayes.classify_many(x_test_data_b)
# acc_b = accuracy_score(x_test_scores_b, pred_b)
# prec_b = precision_score(x_test_scores_b, pred_b, average='micro', zero_division=0)
# rec_b = recall_score(x_test_scores_b, pred_b, average='micro', zero_division=0)
#
# print_task_header("Balanced dataset")
# print(f"class count after balancing:\n{c}")
# print(f"accuracy: {round(acc_b * 100, 2)}%")
# print(f"precision: {round(prec_b * 100, 2)}%")
# print(f"recall: {round(rec_b * 100, 2)}%")

# For a given review (in the form of a list or set of tokens), create a
# dictionary which tells us which words are present and which are not.
# def get_review_features(review, most_common_review_words):
#     review_words = set(review)
#
#     count_most_common = sum({word: word in review_words for word in words_map}.values())
#
#     features = {
#         'length': len(review),
#         'most_common': count_most_common
#     }
#     for index, score in enumerate(scores):
#         key = f'most_common_score_{str(score)}'
#         current = most_common_review_words[index]
#         value = sum({word: word in review_words for word in current}.values())
#         features[key] = value / len(current)
#
#     # features = {word: word in review_words for word in words_map}
#     # features['len'] = len(review)
#     return features


# # %%
# books_data_raw = pd.read_csv('../data/Books_rating.csv')
#
# # %%
# len(books_data_raw)
#
# # %%
# books_data_base = books_data_raw[['review/score', 'review/text']].copy()
# books_data_base.rename(columns={'review/score': 'score', 'review/text': 'review'}, inplace=True)
#
# books_data = books_data.head(100000)

