import numpy as np
import json
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.layers import Input, Conv1D, MaxPooling1D, Dense, Embedding, Flatten, LSTM, Dropout,  Activation
from keras.models import Model, Sequential
from keras.utils import to_categorical
from tokenize_ import word_dict

MAX_NB_WORDS = 50
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM =  300


def label_encoder(Y_):
    return [all_labels.index(i) for i in Y_]


def data_tokenizer(X_, Y_):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_)
    sequences = tokenizer.texts_to_sequences(X_)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = to_categorical(np.asarray(np.asarray(Y_)))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = word_dict.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return data, labels, embedding_matrix, word_index


def create_conv_model(word_index = word_index, embedding_matrix = embedding_matrix):
    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    # x = Conv1D(128, 5, activation='relu')(x)
    # x = MaxPooling1D(35)(x)  # global max pooling
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(len(all_labels), activation='softmax')(x)
    model = Model(sequence_input, preds)
    return model

def LSTM_model(word_index = word_index, embedding_matrix = embedding_matrix):
    model = Sequential()
    model.add(Embedding(len(word_index) + 1,
                        EMBEDDING_DIM,
                        weights=[embedding_matrix],
                        input_length=MAX_SEQUENCE_LENGTH,
                        trainable=False
                        ))
    model.add(LSTM(300))
    model.add(Dense(18, activation='sigmoid'))
    return model


def main():
    train_data = json.load(open('train_data.json'))
    test_data = json.load(open('test_data.json'))
    X,  Y = [x['sentence'] for x in train_data], [x['intent'] for x in train_data]
    X_test,  Y_test = [x['sentence'] for x in test_data], [x['intent'] for x in test_data]
    X_train, X_val, Y_train, Y_val = train_test_split(X,Y,test_size=0.15)

    all_labels = sorted(list(set(Y).union(Y_test)))
    # label_indexer = {k: v for v, k in enumerate(all_labels)}
    Y_train, Y_val, Y_test = label_encoder(Y_train),label_encoder(Y_val), label_encoder(Y_test)

    train_data, train_labels, embedding_matrix, word_index = data_tokenizer(X_train, Y_train)
    test_data, test_labels, _, _ = data_tokenizer(X_test, Y_test)
    val_data, val_labels, _, _ = data_tokenizer(X_val, Y_val)

    conv_model = create_conv_model()

    conv_model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    print(conv_model.summary())

    conv_model.fit(train_data, train_labels,
              validation_data=(val_data, val_labels), epochs=2, batch_size=128
              )


    lstm_model = LSTM_model()
    lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    print(lstm_model.summary())
    lstm_model.fit(train_data, train_labels, epochs=3,
              validation_data=(val_data, val_labels),
              batch_size=64
              )

if __name__ == '__main__':
    main()
