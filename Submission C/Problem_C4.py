# =====================================================================================================
# PROBLEM C4 
#
# Build and train a classifier for the sarcasm dataset. 
# The classifier should have a final layer with 1 neuron activated by sigmoid.
# 
# Do not use lambda layers in your model.
# 
# Dataset used in this problem is built by Rishabh Misra (https://rishabhmisra.github.io/publications).
#
# Desired accuracy and validation_accuracy > 75%
# =======================================================================================================

import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc') is not None and logs.get('acc') > 0.80 and logs.get('val_acc') > 0.80):
      print("\nReached 80% accuracy so cancelling training!")
      self.model.stop_training = True

def solution_C4():
    data_url = 'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/sarcasm.json'
    urllib.request.urlretrieve(data_url, 'sarcasm.json')

    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []

    # YOUR CODE HERE
    datastore = []
    for line in open('./sarcasm.json', 'r'):
        datastore.append(json.loads(line))

    for item in datastore:
        sentences.append(item['headline'])
        labels.append(item['is_sarcastic'])

    train_sentences = sentences[0:training_size]
    test_sentences = sentences[training_size:]
    train_labels = labels[0:training_size]
    test_labels = labels[training_size:]

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(train_sentences)

    word_index = tokenizer.word_index

    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    train_padded = pad_sequences(
        train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    test_sequences = tokenizer.texts_to_sequences(test_sentences)
    test_padded = pad_sequences(
        test_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    train_padded = np.array(train_padded)
    train_labels = np.array(train_labels)
    test_padded = np.array(test_padded)
    test_labels = np.array(test_labels)

    callbacks = myCallback()

    model = tf.keras.Sequential([
    # YOUR CODE HERE. DO not change the last layer or test may fail
        tf.keras.layers.Embedding(
            vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['acc'])

    model.fit(train_padded,
              train_labels,
              epochs=20,
              validation_data=(test_padded, test_labels),
              callbacks=callbacks,
              verbose=1)

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    model = solution_C4()
    model.save("model_C4.h5")
