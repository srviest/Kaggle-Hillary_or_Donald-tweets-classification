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
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.embedding_ops import embedding
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell, lstm
from tflearn.layers.estimator import regression
from sklearn.preprocessing import LabelEncoder

label = {'realDonaldTrump':0, 'HillaryClinton':1}

def read_csv(file_name):
    X,y = [], []
    with open(file_name) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            X.append(row['tweet'])
            y.append(label[row['handle']])
    return X, y

def tokenize(text):
    tokens = []
    stopwords = set(nltk.corpus.stopwords.words('english'))
    toks = nltk.word_tokenize(text.decode('utf-8'))
    tokens += [tok.lower() for tok in toks if tok.lower() not in stopwords and re.match(r'\w', tok)]
    return tokens

def data_preprocess(X):
    all_words = []
    trainX = []
    for x in X:
        all_words+=tokenize(x)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit(all_words)
    for x in X:
        trainX.append(integer_encoded.transform(tokenize(x)))
    return trainX


def build_model():
    # Network building
    net = input_data(shape=[None, 100])
    net = embedding(net, input_dim=10000, output_dim=128)
    net = lstm(net, 128, dropout=0.8)
    # net = dropout(net, 0.5)
    net = fully_connected(net, 2, activation='softmax')
    net = regression(net, optimizer='adam', learning_rate=0.005, loss='categorical_crossentropy')
    model = tflearn.DNN(net, clip_gradients=0., tensorboard_verbose=3, tensorboard_dir=model_dir)
    return model

if __name__ == '__main__':
    file_name = './train.csv'
    X, y = read_csv(file_name)
    trainX = data_preprocess(X)

    # IMDB Dataset loading
    # train, test, _ = imdb.load_data(path='imdb.pkl', n_words=10000,
    #                                 valid_portion=0.1)
    # trainX, trainY = train
    # testX, testY = test

    # Data preprocessing
    # Sequence padding
    trainX = pad_sequences(trainX, maxlen=100, value=0.)
    # testX = pad_sequences(testX, maxlen=100, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(y, 2)
    # testY = to_categorical(testY, 2)


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
    model = build_model()
    model.fit(trainX, trainY, validation_set=0.1, n_epoch=1, show_metric=True, batch_size=64, run_id='test')
    model.save(model_path)
    # print(model.evaluate(testX, testY, batch_size=64))