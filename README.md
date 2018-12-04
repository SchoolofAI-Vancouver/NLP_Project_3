# Natural Language Processing Part 3

organised by [Vancouver School of AI](https://www.meetup.com/Vancouver-School-of-AI/)

**Date: 4 December 2018**

## Project Overview

Build a classification model that can distinguish between toxic and non-toxic comments and use the model in a real-life application.

The meetups serve as guidance. The goal is for all attendees to build a good machine learning model that can be used in a real-life application. We encourage all attendees to apply creativity to this project. There are no limits.

## Installation Requirements

All code is written in Python. Please use [this guide](http://nbviewer.jupyter.org/github/johannesgiorgis/school_of_ai_vancouver/blob/master/intro_to_data_science_tools/01_introduction_to_conda_and_jupyter_notebooks.ipynb) to get Python and Jupyter Notebook up and running.

## Project Setup

### Training 

For those who want to walk though the whole process from training to deployment, you need to download the data to train the model

To download the data, run:
```python
python ml_model/download.py
```
This will download the training data and pre-trained embedding file in :
  - ./assets/data/train.csv
  - ./assets/embedding/fasttext-crawl-300d-2m/crawl-300d-2M.vec
  
To train the model, run:
```python
python ml_model/train_classifier.py
```
This will train a pooled GRU with FastText embedding. The text preprocessor and the model will be seriallized and stored in:
  - ./assets/model/preprocessor.pkl
  - ./assets/model/model.h5
*Note*: This took ~1 hour to train on Intel Core i7-HQ CPU

### Prediction
To get a feeling of doing predictions, run:
```python
python ml_model/predict.py

# output:
# Corgi is stupid          - Toxicity: [0.99293655]
# good boy                 - Toxicity: [0.02075008]
# School of AI is awesome  - Toxicity: [0.01223523]
# F**K                     - Toxicity: [0.90747666]
```

### Deployment
TODO 

## Meetup Content

[Part 3 Slides](https://docs.google.com/presentation/d/1QGz70Qo8ERZBRoFpAb4npl7hvfJM5hhZ2CfIOxjZbBs/edit?usp=sharing)


## Resources

The project uses data from Kaggle's [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge). The data can be found [here](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data).

If you are struggling with implementing some of the concepts discussed at the meetup, check out [the slides notebook](https://github.com/SchoolofAI-Vancouver/NLP_Project_2/blob/master/src/slides.ipynb) as guidance. There are also many [kernels](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/kernels) specific to the toxic comment challenge that you can refer to get some inspiration or help.

Alternatively, ask for assistance on Slack. That's what this community is all about :)

## Code Challenge

**Due Date:** Sunday, 2 December 2018 (PT)

**Challenge:** Build a classification model on the Toxic Comment Classification Challenge [data](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data). You are free to think outside the box here. It does not have to be a binary classifier. Use the techniques discussed in the meetup as a starting point.

**Everyone is encouraged to participate!** The goal of creating models is to have something to deploy as part of a web app (which will be discussed at the next meetup).

The winning solution should ideally contain:

* a well-documented EDA

* machine learning techniques that are sensible

To submit, post your submission's repository link on the `# code_challenge` Slack channel (on the Vancouver School of AI workspace) before the due date.

## Meetup Contributors

[Akshi Chaudhary](https://github.com/akshi8)

[Johannes Harmse](https://github.com/johannesharmse)

[Xinbin Huang](https://github.com/xinbinhuang)

[Peter Lin](https://github.com/peter0083)

[Johannes Giorgis](https://github.com/johannesgiorgis)

[Guru Shiva](https://github.com/gurushiva)

