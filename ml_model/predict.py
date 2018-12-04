#! /usr/bin/env python
# author: Xinbin Huang - Vancouver School of AI
# date: Dec. 3, 2018

import os
from utils import get_root, load_pipeline


ROOT = get_root()
MODEL_PATH = os.path.join(ROOT, 'assets', 'model')
PREPROCESSOR_FILE = os.path.join(MODEL_PATH, 'preprocessor.pkl')
ARCHITECTURE_FILE = os.path.join(MODEL_PATH, 'gru_architecture.json')
WEIGHTS_FILE = os.path.join(MODEL_PATH, 'gru_weights.h5')


class PredictionPipeline(object):

    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model

    def predict(self, text):
        features = self.preprocessor.transform_texts(text)
        pred = self.model.predict(features)
        return pred


if __name__ == "__main__":
    ppl = PredictionPipeline(*load_pipeline(PREPROCESSOR_FILE,
                                            ARCHITECTURE_FILE,
                                            WEIGHTS_FILE))

    sample_text = ['Corgi is stupid',
                   'good boy',
                   'School of AI is awesome',
                   'F**K']

    for text, toxicity in zip(sample_text, ppl.predict(sample_text)):
        print(f"{text}".ljust(25) + f"- Toxicity: {toxicity}")
