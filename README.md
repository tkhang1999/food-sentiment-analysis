# Sentiment Analysis
 
## Introduction
This project is to develop sentiment analysis models for food review using the YELP restaurant review dataset from Kaggle. Due to file size limit on GitHub and our computational limitation, we only provide a small dataset of 10,000 reviews in this repository. For better training and testing, large Yelp dataset, available online, can be used. 

* For the binary classification, we have 2 models: Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) (with and without GloVe embeddings)

* For the fine-grained classification, we also have 2 models: Long Short-Term Memory (LSTM) (with GloVe embeddings) and BERT

## How to run?
* Require `python 3.6` or `python 3.7`

* Run `pip install -r requirements.txt` to install all dependencies for this project.

* Run `python binary_classification/<model_name>.py` or `python fine-grained_classification/<model_name>.py` to see the results.
