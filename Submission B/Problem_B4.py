# ===================================================================================================
# PROBLEM B4
#
# Build and train a classifier for the BBC-text dataset.
# This is a multiclass classification problem.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in: http://mlg.ucd.ie/datasets/bbc.html.
#
# Desired accuracy and validation_accuracy > 91%
# ===================================================================================================

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc') is not None and logs.get('acc') > 0.93 and logs.get('val_acc') > 0.93):
      print("\nReached 92% accuracy so cancelling training!")
      self.model.stop_training = True

# def train_test_split(data, train_size):
#     train = data[:train_size]
#     test = data[train_size:]
#     return train, test

def solution_B4():
    bbc = pd.read_csv('https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/bbc-text.csv')
    
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_portion = .8

    # YOUR CODE HERE
    train_size = int(len(bbc) * training_portion)

    train_cat = bbc['category'][:train_size]
    test_cat = bbc['category'][train_size:]
    train_text = bbc['text'][:train_size]
    test_text = bbc['text'][train_size:]

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(train_text)

    train_sequences = tokenizer.texts_to_sequences(train_text)
    train_padded = pad_sequences(train_sequences, truncating=trunc_type, padding=padding_type, maxlen=max_length)

    validation_sequences = tokenizer.texts_to_sequences(test_text)
    validation_padded = pad_sequences(validation_sequences, truncating=trunc_type, padding=padding_type, maxlen=max_length)

    encoder = LabelEncoder()
    encoder.fit(train_cat)

    y_train = encoder.transform(train_cat)
    y_test = encoder.transform(test_cat)

    num_classes = np.max(y_train) + 1
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    callbacks = myCallback()

    model = tf.keras.Sequential([
        # YOUR CODE HERE.
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Flatten(),
        # tf.keras.layers.Dense(embedding_dim,input_shape=(vocab_size,), activation='relu'),
        tf.keras.layers.Dense(36, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.fit(train_padded, y_train,
              epochs=50,
              batch_size=4,
              validation_data=(validation_padded, y_test),
              validation_split=0.1,
              callbacks=callbacks,
              verbose=1)

    return model

    # The code below is to save your model as a .h5 file.
    # It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    model = solution_B4()
    model.save("model_B4.h5")
