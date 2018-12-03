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

features = train["comment_text"].fillna("# #").values
target = train[classes].values != np.zeros((len(train), 6))
target = target.any(axis=1)
del train
gc.collect()

max_features = 30000
maxlen = 100
embed_size = 300

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(features))
features = tokenizer.texts_to_sequences(features)
features = sequence.pad_sequences(features, maxlen=maxlen)


def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore'))

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: 
        embedding_matrix[i] = embedding_vector


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


def get_model():
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(80, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(1, activation="sigmoid")(conc)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


if __name__ == "__main__":

    model = get_model()

    batch_size = 32
    epochs = 2

    X_train, X_val, y_train, y_val = train_test_split(features, target, train_size=0.95, random_state=233)
    RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)

    hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),
                    callbacks=[RocAuc], verbose=2)

    model.save('../assets/model/model.h5')
