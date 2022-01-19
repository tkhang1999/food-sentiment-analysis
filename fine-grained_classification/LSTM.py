import gzip
import shutil
import os
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

from tensorflow.python.keras.layers import Dense, Embedding, LSTM, Dropout, Input, Bidirectional, GlobalMaxPool1D
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences 
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.models import Model


'''
Use tensorflow 1.15 for training
'''

print('Training using tensorflow version %s' % tf.__version__)
tf.get_logger().setLevel('ERROR')
SEED = 42
np.random.seed(SEED)
tf.set_random_seed(SEED)

###########################################################

dataset = pd.read_csv('./data/yelp.csv')
print(dataset['stars'].value_counts())

dataset['stars'] = dataset['stars'] - 1
X_train, X_test, y_train, y_test = train_test_split(dataset['text'], dataset['stars'], test_size=0.1, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1/9, random_state=SEED) #0.9/9=0.1

# Comment this if do binary classification
y_train = tf.keras.utils.to_categorical(y_train)
y_val = tf.keras.utils.to_categorical(y_val)
y_test = tf.keras.utils.to_categorical(y_test)

print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)

###########################################################

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 30000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 250
# This is fixed.
EMBEDDING_DIM = 50
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(dataset['text'])
word_index = tokenizer.word_index
if (len(word_index) < MAX_NB_WORDS):
    raise Exception('Found %s unique tokens. MUST greater than MAX_NB_WORDS = %s. Set MAX_NB_WORDS to a lower number' % (len(word_index), MAX_NB_WORDS))
print('Found %s unique tokens' % len(word_index))


x_train_tokens = tokenizer.texts_to_sequences(X_train)
x_val_tokens = tokenizer.texts_to_sequences(X_val)
x_test_tokens = tokenizer.texts_to_sequences(X_test)
num_tokens = [len(tokens) for tokens in x_train_tokens + x_val_tokens + x_test_tokens]
num_tokens = np.array(num_tokens)

max_tokens = int(np.mean(num_tokens)+2*np.std(num_tokens))

x_train = pad_sequences(x_train_tokens, maxlen = MAX_SEQUENCE_LENGTH)
x_val = pad_sequences(x_val_tokens, maxlen = MAX_SEQUENCE_LENGTH)
x_test = pad_sequences(x_test_tokens, maxlen=MAX_SEQUENCE_LENGTH)

###########################################################

EMBEDDING_FILE_ZIP = './data/glove.6B.50d.txt.gz'
EMBEDDING_FILE = './data/glove.6B.50d.txt'

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

###########################################################

inp = Input(shape=(MAX_SEQUENCE_LENGTH,))
x = Embedding(MAX_NB_WORDS, EMBEDDING_DIM, weights=[embedding_matrix])(inp)
x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
x = GlobalMaxPool1D()(x)
x = Dense(50, activation='relu')(x)
x = Dropout(0.1)(x)
x = Dense(5, activation='softmax')(x)
lstm_model = Model(inputs=inp, outputs=x)
lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(lstm_model.summary())

epochs = 5
batch_size = 128

history = lstm_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,validation_data=(x_val, y_val),callbacks=[EarlyStopping(monitor='val_loss',patience=1, min_delta=0.0001)])
result = lstm_model.evaluate(x_test,y_test)
print(result)
