#! /usr/bin/env python
# author: Xinbin Huang - Vancouver School of AI
# date: Dec. 3, 2018
# Usage:
#    python train_classifier.py


import gc
import os
import pickle
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import TensorBoard

from utils import get_logger, get_root


np.random.seed(42)
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '4'

DIR_ROOT = get_root()
DIR_ASSETS = os.path.join(DIR_ROOT, 'assets')
MODEL_PATH = os.path.join(DIR_ASSETS, 'model')
LOG_PATH = os.path.join(DIR_ASSETS, 'tb_logs')
EMBEDDING_FILE = os.path.join(DIR_ASSETS, 'embedding', 'fasttext-crawl-300d-2m', 'crawl-300d-2M.vec')
DATA_FILE = os.path.join(DIR_ASSETS, 'data', 'train.csv')

MAX_FEATURES = 30000
MAXLEN = 100
EMBED_SIZE = 300
TRAIN_SIZE = 0.95
BATCH_SIZE = 32
EPOCHS = 2


def convert_binary_toxic(data, classes):
    target = data[classes].values != np.zeros((len(data), 6))
    binary = target.any(axis=1)
    return binary


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


def get_embeddings(embed_file, word_index, max_features, embed_size):

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

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


class RocAucEvaluation(TensorBoard):

    def __init__(self, log_dir='./logs',
                 histogram_freq=0,
                 batch_size=32,
                 write_graph=True,
                 write_grads=False,
                 write_images=False,
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None,
                 embeddings_data=None,
                 update_freq='epoch',
                 validation_data=(),
                 interval=1):
        super().__init__(log_dir=log_dir, batch_size=batch_size)

        self.X_val, self.y_val = validation_data
        self.interval = interval

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)

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

    logger = get_logger()

    logger.info(f"Loading data: {DATA_FILE}")
    train = pd.read_csv(DATA_FILE)
    classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    features = train["comment_text"].fillna("# #").values
    target = convert_binary_toxic(train, classes)
    del train
    gc.collect()

    logger.info(f"Transforming data")
    preprocessor = Preprocess(max_features=MAX_FEATURES, maxlen=MAXLEN)
    preprocessor.fit_texts(list(features))
    features = preprocessor.transform_texts(features)
    word_index = preprocessor.tokenizer.word_index

    PRERPOCESSOR_FILE = os.path.join(MODEL_PATH, 'preprocessor.pkl')
    logger.info(f"Saving the text transformer: {PRERPOCESSOR_FILE}")

    with open(PRERPOCESSOR_FILE, 'wb') as file:
        pickle.dump(preprocessor, file)
    del preprocessor
    gc.collect()

    logger.info(f"Loading embedding vectors: {EMBEDDING_FILE}")
    embedding_matrix = get_embeddings(EMBEDDING_FILE, word_index, MAX_FEATURES, EMBED_SIZE)

    logger.info(f"Model training, train size: {TRAIN_SIZE}")
    X_train, X_val, y_train, y_val = train_test_split(features, target, train_size=TRAIN_SIZE, random_state=233)
    RocAuc = RocAucEvaluation(log_dir=LOG_PATH, batch_size=BATCH_SIZE, validation_data=(X_val, y_val), interval=1)

    model = get_model(MAXLEN, MAX_FEATURES, EMBED_SIZE, embedding_matrix)

    hist = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                     validation_data=(X_val, y_val), callbacks=[RocAuc], verbose=1)

    ARCHITECTURE_FILE = os.path.join(MODEL_PATH, 'gru_architecture.json')
    logger.info(f"Saving the architecture: {ARCHITECTURE_FILE}")

    with open(ARCHITECTURE_FILE, 'w') as file:
        architecture_json = model.to_json()
        file.write(architecture_json)

    WEIGHTS_FILE = os.path.join(MODEL_PATH, 'gru_weights.h5')
    logger.info(f"Saving the weights: {WEIGHTS_FILE}")

    model.save_weights(WEIGHTS_FILE)
