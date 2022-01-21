import gzip
import os
import shutil
import pandas as pd
import numpy as np 

from sklearn.model_selection import train_test_split 
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Embedding, LSTM, Dropout, Input, Bidirectional, GlobalMaxPool1D
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences 
from tensorflow.python.keras.optimizers import Adam

# Data pre-processing
dataset = pd.read_csv('./data/yelp.csv')
sentences = dataset['text'].values
y = dataset['label'].values

# Prepare train and test data
x_train, x_test, y_train, y_test = train_test_split(dataset['text'], dataset['label'], test_size=0.1, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=1/9, random_state=42) #0.9/9=0.1

num_words=5000

embedding_size=32
optimizer = Adam(lr=1e-3)

tokenizer = Tokenizer(num_words = num_words)
tokenizer.fit_on_texts(sentences)
vocab_size=len(tokenizer.word_index)+1
print(vocab_size)

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

# Create the default model
default_model = Sequential()
default_model.add(Embedding(input_dim = vocab_size,output_dim=embedding_size,input_length=max_tokens))
default_model.add(LSTM(100))
default_model.add(Dense(1,activation='sigmoid'))
print(default_model.summary())

# Train and Test default model
default_model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
default_model.fit(x_train,y_train,epochs=4,batch_size=128,validation_data=(x_val, y_val))
print('finish training default model')
result = default_model.evaluate(x_test,y_test)
print(result)

# load embedding from file
EMBEDDING_FILE_ZIP = './data/glove.6B.50d.txt.gz'
EMBEDDING_FILE = './data/glove.6B.50d.txt'
EMBEDDING_DIM = 50
MAX_NB_WORDS = 30000

if not os.path.isfile(EMBEDDING_FILE):
    print('Extracting ' + EMBEDDING_FILE_ZIP)
    with gzip.open(EMBEDDING_FILE_ZIP, 'rb') as f_in:
        with open(EMBEDDING_FILE, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE, encoding='utf8'))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
emb_mean,emb_std

word_index = tokenizer.word_index
nb_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

# Create the custom model
inp = Input(shape=(max_tokens,))
x = Embedding(MAX_NB_WORDS, EMBEDDING_DIM, weights=[embedding_matrix])(inp)
x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
x = GlobalMaxPool1D()(x)
x = Dense(50, activation='relu')(x)
x = Dropout(0.1)(x)
x = Dense(1, activation='sigmoid')(x)
custom_model = Model(inputs=inp, outputs=x)
custom_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train and Test custom model
custom_model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
custom_model.fit(x_train,y_train,epochs=5,batch_size=128,validation_data=(x_val, y_val))
print('finish training custom model')
result = custom_model.evaluate(x_test,y_test)
print(result)
