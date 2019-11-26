import re
import tensorflow as tf
import numpy as np
import os
import pandas as pd
import tensorflow_hub as hub
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense
from tensorflow.keras.preprocessing import sequence


# Load all files from a directory in a DataFrame.
def load_directory_data(directory):
    data = {"sentence": []}
    for file_path in os.listdir(directory):
        with open(directory + "/" + file_path) as f:
            text = f.read()
            words = set(tf.keras.preprocessing.text.text_to_word_sequence(text))
            vocab_size = len(words)
            result = tf.keras.preprocessing.text.one_hot(text, round(vocab_size * 1.3))
            data['sentence'].append(result)

    return pd.DataFrame.from_dict(data)


# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset():
    pos_df = load_directory_data('Data/train/pos')
    neg_df = load_directory_data('Data/train/neg')
    pos_df["polarity"] = 1
    neg_df["polarity"] = 0
    return pd.concat([pos_df, neg_df])


x_train = load_dataset()['sentence'].values
y_train = load_dataset()['polarity'].values

x_train = sequence.pad_sequences(x_train, maxlen=2000)
y_train = utils.to_categorical(y_train, 2)

model = Sequential()
model.add(Embedding(2000, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train,y_train,epochs=1)

model.save('model.h5')

