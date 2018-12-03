#! /usr/bin/env python
# author: Xinbin Huang - Vancouver School of AI
# date: Dec. 3, 2018
#
# Partially borrowed from: https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e
# This script will download the required training data and embedding file in the project
#      - ./assets/data/train.csv
#      - ./assets/embedding/fasttext-crawl-300d-2m/crawl-300d-2M.vec
# Usage:
#    python download.py


import os
import urllib.request
import zipfile

from utils import get_root


DIR_ROOT = get_root()
DIR_ASSETS = os.path.join(DIR_ROOT, 'assets')
DATA_DIR = os.path.join(DIR_ASSETS, 'data')
EMBEDDING_DIR = os.path.join(DIR_ASSETS, 'embedding', 'fasttext-crawl-300d-2m')


TASKS = ['TrainData', 'Embedding']
TASK2PATH = {'TrainData': 'https://storage.googleapis.com/kaggle-competitions-data/kaggle/8076/train.csv.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1544128430&Signature=TGKJMbm2U4xUmsXVNcXuymTIgtuq8LBkrqlmc%2F8H%2BQ06DYm5IevQv9OaZAkKuKey1KYQ6HKjfVIoXWKOQLkN0IZxz1YVH6Vf81EvweVys%2Fvu0NZ33UkZ73iT%2BTkWaKpsOQSfiEQZIPEI%2FEECb7Tvlaj8Nx1l0Ozgh9hSC8juIltL2b%2F5z9nEMpHajnHrHT4zWkt%2BeFUL%2FrvhjrspenB7TBUqoQY9aB74OYZ3K9P4f17d8r5cKOJ7g510eEkrVu7aNYP1U6zueIstjpT1bS5obBDcIWc3rr2KJMj2qabHVzel9%2BWXq0%2BOItH68X0N%2BKqxxJYPiE%2FUnxYqskYNmrWfOg%3D%3D',
             'Embedding': 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M.vec.zip'}
TASK2DIR = {'TrainData': DATA_DIR,
            'Embedding': EMBEDDING_DIR}


def download_and_extract(task, task_url, data_dir):
    print(f"Downloading and extracting {task}")
    data_file = f"{task}.zip"
    urllib.request.urlretrieve(task_url, data_file)
    with zipfile.ZipFile(data_file) as zip_ref:
        zip_ref.extractall(data_dir)
    os.remove(data_file)
    print("\tCompleted")


def main():
    for task in TASKS:
        task_dir = TASK2DIR[task]
        task_url = TASK2PATH[task]
        if not os.path.isdir(task_dir):
            os.mkdir(task_dir)
        download_and_extract(task, task_url, task_dir)


if __name__ == "__main__":
    main()
