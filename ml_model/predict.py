#! /usr/bin/env python
# author: Xinbin Huang - Vancouver School of AI
# date: Dec. 3, 2018


# modules
import os
import sys

# add current directory to sys.path
# needed for Flask app to work
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

# custom modules
from utils import get_root, load_pipeline, get_logger


ROOT = get_root()
MODEL_PATH = os.path.join(ROOT, 'assets', 'model')
PREPROCESSOR_FILE = os.path.join(MODEL_PATH, 'preprocessor.pkl')
ARCHITECTURE_FILE = os.path.join(MODEL_PATH, 'gru_architecture.json')
WEIGHTS_FILE = os.path.join(MODEL_PATH, 'gru_weights.h5')

print(f"Root Directory: {ROOT_DIR}")
print(f"Model Path: {MODEL_PATH}")
print(f"Preprocessor file: {PREPROCESSOR_FILE}")
print(f"Architecture File: {ARCHITECTURE_FILE}")
print(f"Weights File: {WEIGHTS_FILE}")


class PredictionPipeline(object):

    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model

    def predict(self, text):
        features = self.preprocessor.transform_texts(text)
        pred = self.model.predict(features)
        return pred



if __name__ == "__main__":

    logger = get_logger()
    logger.info("Script Started")
    logger.info("Loading model...")
    ppl = PredictionPipeline(*load_pipeline(PREPROCESSOR_FILE, 
                                            ARCHITECTURE_FILE, 
                                            WEIGHTS_FILE))
    logger.info("Completed loading model!")

    sample_text = ['Corgi is stupid',
                   'good boy',
                   'School of AI is awesome',
                   'F**K']

    for text, toxicity in zip(sample_text, ppl.predict(sample_text)):
        print(f"{text}".ljust(25) + f"- Toxicity: {toxicity}")

    logger.info("Script Completed")
