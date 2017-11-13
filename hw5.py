# -*- coding: utf-8 -*-
"""
Simple example using LSTM recurrent neural network to classify IMDB
sentiment dataset.
References:
    - Long Short Term Memory, Sepp Hochreiter & Jurgen Schmidhuber, Neural
    Computation 9(8): 1735-1780, 1997.
    - Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng,
    and Christopher Potts. (2011). Learning Word Vectors for Sentiment
    Analysis. The 49th Annual Meeting of the Association for Computational
    Linguistics (ACL 2011).
Links:
    - http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
    - http://ai.stanford.edu/~amaas/data/sentiment/
"""

from __future__ import division, print_function, absolute_import
import csv, os, errno, nltk, re, math
import tflearn
import numpy as np

from tflearn.data_utils import to_categorical
from tflearn.datasets import imdb
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.embedding_ops import embedding
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell, lstm
from tflearn.layers.estimator import regression

from keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D, Dropout, Activation
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

label = {'realDonaldTrump':0, 'HillaryClinton':1, 'none':-1}

def read_csv(file_name):
    X,y = [], []
    with open(file_name) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            X.append(row['tweet'])
            y.append(label[row['handle']])
    return (X,y)

def make_dictionary(trainX, dictionary_size=10000):
    tokenizer = Tokenizer(num_words=dictionary_size)
    tokenizer.fit_on_texts(trainX)
    return tokenizer


def build_model_tflearn():
    # Network building
    net = input_data(shape=[None, 100])
    net = embedding(net, input_dim=dict_size, output_dim=128)
    net = lstm(net, 128, dropout=0.8)
    # net = dropout(net, 0.5)
    net = fully_connected(net, 2, activation='softmax')
    net = regression(net, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy')
    model = tflearn.DNN(net, clip_gradients=0., tensorboard_verbose=3, tensorboard_dir=model_dir)
    return model

def build_model_crnn():
    model = Sequential()
    model.add(Embedding(10000, 128, input_length=200))
    model.add(Dropout(0.2))
    model.add(Conv1D(64, 5, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(LSTM(128))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    train_file = './train.csv'
    test_file = './test.csv'
    (train_data, trainY) = read_csv(train_file)
    (test_data, testY) = read_csv(test_file)

    tokenizer = make_dictionary(train_data)
    trainX = tokenizer.texts_to_sequences(train_data)
    testX = tokenizer.texts_to_sequences(test_data)

    # Sequence padding
    trainX = pad_sequences(trainX, maxlen=200)
    testX = pad_sequences(testX, maxlen=200)
    
    # make save folder
    model_dir = './model'
    model_path = os.path.join(model_dir, 'lstm.pth')
    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Directory already exists.')
        else:
            raise

    # Training
    model = build_model_crnn()
    print(trainX.shape, len(trainY))
    model.fit(trainX, trainY, validation_split=0.1, epochs=10)
    # model.fit(trainX, trainY, validation_set=0.1, n_epoch=20, show_metric=True, batch_size=32, run_id='test')
    model.save(model_path)
    # print(model.evaluate(testX, testY, batch_size=64))
    predict = model.predict(testX)
    np.savetxt(predict, './test_predict.csv')
