#! /usr/bin/env python

import numpy as np
np.random.seed(42)
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback

import warnings
warnings.filterwarnings('ignore')

import gc
import os
os.environ['OMP_NUM_THREADS'] = '4'


EMBEDDING_FILE = '../assets/embedding/fasttext-crawl-300d-2m/crawl-300d-2M.vec'
train = pd.read_csv('../assets/data/train.csv')

classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def convert_binary_toxic(data, classes):
    target = data[classes].values != np.zeros((len(data), 6))
    binary = target.any(axis=1)
    return binary


target = convert_binary_toxic(train, classes)
features = train["comment_text"].fillna("# #").values
del train
gc.collect()

MAX_FEATURES = 30000
MAXLEN = 100
EMBED_SIZE = 300


class Preprocess(object):

    def __init__(self, max_features, maxlen):
        self.max_features = max_features
        self.maxlen = maxlen

    def fit_texts(self, list_sentences):
        self.tokenizer = text.Tokenizer(num_words=self.max_features)
        self.tokenizer.fit_on_texts(list_sentences)

    def transform_texts(self, list_sentences):
        tokenized_sentences = self.tokenizer.texts_to_sequences(list_sentences)
        features = sequence.pad_sequences(tokenized_sentences, maxlen=self.maxlen)
        return features


pre = Preprocess(max_features=MAX_FEATURES, maxlen=MAXLEN)
pre.fit_texts(list(features))
features = pre.transform_texts(features)


def get_embeddings(embed_file, word_index, max_features, embed_size):
    def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
    embeddings_pretrained = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(embed_file, encoding="utf8", errors='ignore'))
 
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.zeros((nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_pretrained.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    
    return embedding_matrix


word_index = pre.tokenizer.word_index
embedding_matrix = get_embeddings(EMBEDDING_FILE, word_index, MAX_FEATURES, EMBED_SIZE)


class RocAucEvaluation(Callback):

    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.X_val, self.y_val = validation_data
        self.interval = interval

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))


def get_model(maxlen, max_features, embed_size, embedding_matrix):
    input = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(input)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(80, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    output = Dense(1, activation="sigmoid")(conc)

    model = Model(inputs=input, outputs=output)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


if __name__ == "__main__":

    model = get_model()

    BATCH_SIZE = 32
    EPOCHS = 2

    X_train, X_val, y_train, y_val = train_test_split(features, target, train_size=0.95, random_state=233)
    RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)

    hist = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, y_val),
                    callbacks=[RocAuc], verbose=1)

    model.save('../assets/model/model.h5')
