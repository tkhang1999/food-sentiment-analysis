import gzip
import os
import shutil
import numpy as np 
import pandas as pd

from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import Dense, Embedding, Flatten
from tensorflow.python.keras.layers.convolutional import Conv1D, MaxPooling1D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences 

# Load embedding as a dict
def load_embedding(filename):
    # load embedding into memory, skip first line
    file = open(filename,'r', encoding="utf8")
    lines = file.readlines()
    file.close()
    # create a map of words to vectors
    embedding = dict()
    for line in lines:
        parts = line.split()
        # key is string word, value is numpy array for vector
        embedding[parts[0]] = np.asarray(parts[1:], dtype='float32')
    return embedding

# Create a weight matrix for the Embedding layer from a loaded embedding
def get_weight_matrix(embedding, vocab, embedding_dim):
    # total vocabulary size plus 0 for unknown words
    vocab_size = len(vocab) + 1
    # define weight matrix dimensions with all 0
    weight_matrix = np.zeros((vocab_size, embedding_dim))
    # step vocab, store vectors using the Tokenizer's integer mapping
    for word, i in vocab.items():
        vector = embedding.get(word)
        if vector is not None:
            weight_matrix[i] = vector
    return weight_matrix

# Data pre-processing
dataset = pd.read_csv('./data/yelp.csv')
sentences = dataset['text'].values
y = dataset['label'].values

x_train, x_test, y_train, y_test = train_test_split(dataset['text'], dataset['label'], test_size=0.1, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=1/9, random_state=42) #0.9/9=0.1

num_words=5000

embedding_size=32
optimizer = Adam(lr=1e-3)

tokenizer = Tokenizer(num_words = num_words)
tokenizer.fit_on_texts(sentences)
vocab_size=len(tokenizer.word_index)+1
print(vocab_size)

# Prepare train and test data
x_train_tokens = tokenizer.texts_to_sequences(x_train)
x_val_tokens = tokenizer.texts_to_sequences(x_val)
x_test_tokens = tokenizer.texts_to_sequences(x_test)
num_tokens = [len(tokens) for tokens in x_train_tokens + x_val_tokens + x_test_tokens]
num_tokens = np.array(num_tokens)

max_tokens = int(np.mean(num_tokens)+2*np.std(num_tokens))

x_train = pad_sequences(x_train_tokens,padding='pre',maxlen=max_tokens)
x_val = pad_sequences(x_val_tokens,padding='pre',maxlen=max_tokens)
x_test = pad_sequences(x_test_tokens,padding='pre',maxlen=max_tokens)

print(y_train.shape)
print(x_train.shape)

# Create the default model with a Word Embedding layer using test data
default_model = Sequential()
default_model.add(Embedding(input_dim = vocab_size,output_dim=embedding_size,input_length=max_tokens))
default_model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
default_model.add(MaxPooling1D(pool_size=2))
default_model.add(Flatten())
default_model.add(Dense(1, activation='sigmoid'))
print(default_model.summary())

# Train and Test default model
default_model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
default_model.fit(x_train,y_train,epochs=3,batch_size=128,validation_data=(x_val, y_val))
print('finish training default model')

print(default_model.evaluate(x_test,y_test))

# load embedding from file
EMBEDDING_FILE_ZIP = './data/glove.6B.50d.txt.gz'
EMBEDDING_FILE = './data/glove.6B.50d.txt'
EMBEDDING_DIM = 50

if not os.path.isfile(EMBEDDING_FILE):
    print('Extracting ' + EMBEDDING_FILE_ZIP)
    with gzip.open(EMBEDDING_FILE_ZIP, 'rb') as f_in:
        with open(EMBEDDING_FILE, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
raw_embedding = load_embedding(EMBEDDING_FILE)
# get vectors in the right order
embedding_vectors = get_weight_matrix(raw_embedding, tokenizer.word_index, EMBEDDING_DIM)
# create the embedding layer
embedding_layer = Embedding(vocab_size, 50, weights=[embedding_vectors], input_length=max_tokens, trainable=True)

# Create the custom model with a Word Embedding layer using glove.6B data
custom_model = Sequential()
custom_model.add(embedding_layer)
custom_model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
custom_model.add(MaxPooling1D(pool_size=2))
custom_model.add(Flatten())
custom_model.add(Dense(1, activation='sigmoid'))
print(custom_model.summary())

# Train and Test custom model
custom_model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
custom_model.fit(x_train,y_train,epochs=4,batch_size=128,validation_data=(x_val, y_val))
print('finish training custom model')

print(custom_model.evaluate(x_test,y_test))