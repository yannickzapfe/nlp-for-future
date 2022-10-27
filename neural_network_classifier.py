import sys

from keras.layers import Embedding, SpatialDropout1D, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, Dense, \
    Dropout

import preprocessing

from keras import Input, Model
from tcn import TCN

from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

if len(sys.argv) > 1:
    n_first = int(sys.argv[1])
else:
    n_first = 1000

books_data = preprocessing.load_books_rating_data(n_first)

X_tokenized = preprocessing.tokenize(books_data.review)

vocab = set([word for sent in X_tokenized for word in sent])
word_index = {}
index = 1
for word in vocab:
    word_index[word] = index
    index += 1

X_indexed = [[word_index[word] for word in sent] for sent in X_tokenized]
X_padded = preprocessing.nn_pad(X_indexed, None)

y_one_hot = preprocessing.one_hot_encode(books_data.score)

X_train, X_test, y_train_one_hot, y_test_one_hot = train_test_split(X_padded, y_one_hot, test_size=0.20,
                                                                    random_state=42)

embeddings_matrix = preprocessing.create_embeddings_matrix(word_index)
input_length = len(X_padded[0])


def tcn_model(input_length, emb_matrix):
    inp = Input(shape=(input_length,))
    x = Embedding(input_dim=embeddings_matrix.shape[0],
                  output_dim=embeddings_matrix.shape[1],
                  input_length=input_length,
                  # Assign the embedding weight with word2vec embedding marix
                  weights=[emb_matrix],
                  # Set the weight to be not trainable (static)
                  trainable=False)(inp)

    x = SpatialDropout1D(0.1)(x)

    x = TCN(128, dilations=[1, 2, 4], return_sequences=True, activation='relu', name='tcn1')(x)
    x = TCN(64, dilations=[1, 2, 4], return_sequences=True, activation='relu', name='tcn2')(x)

    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)

    conc = concatenate([avg_pool, max_pool])
    conc = Dense(16, activation="relu")(conc)
    conc = Dropout(0.1)(conc)
    outp = Dense(5, activation="softmax")(conc)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


deep_model = tcn_model(input_length, embeddings_matrix)

BATCH_SIZE = 1024
EPOCHS = 8
history = deep_model.fit(X_train, y_train_one_hot, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1,
                         validation_split=0.2)

predictions_one_hot = deep_model.predict(X_test)
predictions = [list(one_hot).index(max(one_hot)) + 1 for one_hot in predictions_one_hot]
y_test = [list(one_hot).index(max(one_hot)) + 1 for one_hot in y_test_one_hot]
print("     MSE:", mean_squared_error(y_test, predictions))
print("Accuracy:", accuracy_score(y_test, predictions))
