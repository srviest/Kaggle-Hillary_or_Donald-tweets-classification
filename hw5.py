# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import csv, os, errno, nltk, re, math, collections
import numpy as np
import argparse
from keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D, Dropout, Activation
from keras.models import Sequential, load_model
from keras.preprocessing.text import Tokenizer
from keras.optimizers import Adam, SGD
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, Callback, ReduceLROnPlateau, TensorBoard
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

parser = argparse.ArgumentParser(description='Homework 5: sentiment analysis')
parser.add_argument('mode', metavar="mode", default="predict",  help='train or predict')
parser.add_argument('--model', metavar="mode", default="crnn",  help='model')
args = parser.parse_args()

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

class History(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.acc = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))

class BuildModel(object):
    def __init__(self, model, optimizer='adam'):
        self.model = model
        print('Initial learning rate: %f'%init_lr)
        if optimizer=='adam':
            self.optimizer = Adam(lr=init_lr)
        elif optimizer=='sgd':
            self.optimizer = SGD(lr=init_lr, momentum=momentum)
    def build(self):        
        if self.model=='crnn':
            return self.build_model_crnn()
        elif self.model=='cnn':
            return self.build_model_cnn()
        elif self.model=='rnn':
            return self.build_model_rnn()

    def build_model_crnn(self):
        model = Sequential()
        model.add(Embedding(dictionary_size, 128, input_length=padding_length))
        model.add(Dropout(0.5))
        model.add(Conv1D(64, 5, activation='relu'))
        model.add(MaxPooling1D(pool_size=4))
        model.add(LSTM(128, recurrent_dropout = 0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        return model

    def build_model_rnn(self):
        model = Sequential()
        model.add(Embedding(dictionary_size, 64, input_length=padding_length))
        model.add(LSTM(128, recurrent_dropout = 0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        return model

    def build_model_cnn(self):
        model = Sequential()
        model.add(Embedding(dictionary_size, 128, input_length=padding_length))
        model.add(Dropout(0.2))
        # model.add(Conv1D(filters,
        #          kernel_size,
        #          padding='valid',
        #          activation='relu',
        #          strides=1))
        # we use max pooling:
        model.add(GlobalMaxPooling1D())

        # We add a vanilla hidden layer:
        model.add(Dense(hidden_dims))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        return model

def write_result(predict, result_path):
    d = 1-predict
    length = predict.shape[0]
    result = np.ndarray((length,3), dtype = object)
    result[:,0] = range(length)
    result[:,1] = d[:,0]
    result[:,2] = predict[:,0]
    np.savetxt(result_path, result, 
        delimiter=',', fmt=['%d', '%f', '%f'], 
        header='id,realDonaldTrump,HillaryClinton', comments='')

def train(model):
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=3, verbose=1, min_lr=0.001)
    tfboard = TensorBoard(log_dir=model_dir, histogram_freq=0, 
        write_graph=True, write_images=True, 
        embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    history = History()
    checkpointer = ModelCheckpoint(filepath=model_dir+'.ckpt', monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False)
    callbacks=[checkpointer, history, reduce_lr]
    print(collections.Counter(trainY))
    model.fit(trainX, trainY, batch_size=batch_num, validation_split=validation_split, epochs=epoch, callbacks=callbacks)
    model.save(model_path)
    np.savetxt(os.path.join(model_dir, 'loss.csv'), history.losses, fmt='%s')
    np.savetxt(os.path.join(model_dir, 'acc.csv'), history.acc, fmt='%s')
    
if __name__ == '__main__':
    # I/O parameters
    train_file = './train.csv'
    test_file = './test.csv'
    model_dir = './model'+'_'+args.model
    model_path = os.path.join(model_dir, args.model+'.hdf5')

    # experiemnt parameters
    validation_split = 0.2
    dictionary_size = 10000
    batch_num = 64
    epoch = 5
    init_lr = 0.01
    momentum = 0.9
    optimizer='adam'
    padding_length=200

    # read data and preprocess
    (train_data, trainY) = read_csv(train_file)
    (test_data, testY) = read_csv(test_file)
    tokenizer = make_dictionary(train_data, dictionary_size=dictionary_size)
    trainX = tokenizer.texts_to_sequences(train_data)
    testX = tokenizer.texts_to_sequences(test_data)

    # Sequence padding
    trainX = pad_sequences(trainX, maxlen=padding_length)
    testX = pad_sequences(testX, maxlen=padding_length)
    
    # make save folder
    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Directory already exists.')
        else:
            raise

    # select model
    model_builder = BuildModel(args.model, optimizer)
    model = model_builder.build()

    # train or predict
    if args.mode=='train':
        print("Start training")
        train(model)
    elif args.mode=='predict':
        # Laod pre-trained model
        if os.path.isfile(model_path):
            print("Predict using pre-trained model")
            model = load_model(model_path)
        # Training
        else:
            print("Unable to find pre-trained model, training from scratch")
            train(model)
        print("Predicting...")
        predict = model.predict(testX)
        write_result(predict, './prediction.csv')
        
