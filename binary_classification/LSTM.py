import pandas as pd
import numpy as np 

from sklearn.model_selection import train_test_split 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Embedding, LSTM
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

# Create the model
model = Sequential()
model.add(Embedding(input_dim = vocab_size,output_dim=embedding_size,input_length=max_tokens))
model.add(LSTM(100))
model.add(Dense(1,activation='sigmoid'))
print(model.summary())

# Train and Test
model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
model.fit(x_train,y_train,epochs=3,batch_size=128,validation_data=(x_val, y_val))
print('finish training')
result = model.evaluate(x_test,y_test)
print(result)