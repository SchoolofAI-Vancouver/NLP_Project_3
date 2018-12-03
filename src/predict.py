#! /usr/bin/env python

import os
import pickle

from keras.models import load_model
from utils import get_root

ROOT = get_root()
MODEL_PATH = os.path.join(ROOT, 'assets', 'model', 'model.h5')
PREPROCESSOR_PATH = os.path.join(ROOT, 'assets', 'model', 'preprocessor.pkl')

model = load_model(MODEL_PATH)
preprocessor = pickle.load(open(PREPROCESSOR_PATH, 'rb'))


def predict(text):
    features = preprocessor.transform_texts(text)
    return model.predict(features)


if __name__ == "__main__":
    text = ["Fuck you idiot!", "good boy!"]
    print(predict(text))