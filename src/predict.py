#! /usr/bin/env python
# author: Xinbin Huang - Vancouver School of AI
# date: Dec. 3, 2018

import os
import pickle

from keras.models import load_model
from utils import get_root


ROOT = get_root()
MODEL_FILE = os.path.join(ROOT, 'assets', 'model', 'model.h5')
PREPROCESSOR_FILE = os.path.join(ROOT, 'assets', 'model', 'preprocessor.pkl')


def load_pipeline(preprocessor_file, model_file):
    preprocessor = pickle.load(open(preprocessor_file, 'rb'))
    model = load_model(model_file)
    return preprocessor, model


class PredictionPipeline(object):

    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model

    def predict(self, text):
        features = self.preprocessor.transform_texts(text)
        pred = self.model.predict(features)
        return pred


if __name__ == "__main__":
    ppl = PredictionPipeline(*load_pipeline(PREPROCESSOR_FILE, MODEL_FILE))

    text = ['you idiot']
    print(ppl.predict(text))
