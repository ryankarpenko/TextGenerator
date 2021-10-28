# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 17:05:36 2020

@author: penko
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pyplot as plt

import numpy as np

import tensorflow as tf

from tensorflow.keras import layers

import re

import pickle

import os

import scipy.special as ss

# Keras LSTM Network for various classic books


# Get holmes text vector
hfile = open(r'C:\Users\penko\OneDrive\ML_Practice\Neural Net Practice\holmes_trim.txt', encoding = 'utf8')

# Get copperfield text vector
hfile = open(r'C:\Users\penko\OneDrive\ML_Practice\Neural Net Practice\copperfield.txt', encoding = 'utf8')

holmes = hfile.read().replace('\n',' ').replace('_','').replace('½','').replace("'", '').replace('"','').replace('‘', '').replace('’', '').replace('&c', 'etc').replace('“', '').replace('”', '').replace('â', 'a').replace('æ', 'ae').replace('è', 'e').replace('é', 'e').replace('œ', 'oe').replace('£', '').replace('à', 'a')

hvec = re.findall(r'\w+', holmes)

#hvec[:100]
#list(set(x for L in hvec for x in L))

# Create input and output
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(hvec)

'''
# saving
with open(r'C:\Users\penko\OneDrive\ML_Practice\Neural Net Practice\tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
close(handle)

# loading
with open(r'C:\Users\penko\OneDrive\ML_Practice\Neural Net Practice\tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

'''

encoded = tokenizer.texts_to_sequences(hvec)

#vocab_size = len(tokenizer.word_index) + 1
vocab_size = len(tokenizer.word_index) + 1

sequences = list()

for i in range(1, len(encoded)):
    sequence = encoded[i-1:i+1]
    sequences.append(sequence)

    
x, y = [ s[0] for s in sequences ], [ s[1] for s in sequences ]

y = tf.keras.utils.to_categorical(y, num_classes = vocab_size, dtype = "int")
x = np.array(x, dtype = "int")

'''
# Generic
x_train, y_train = x[:(4*len(x)//5)], y[:(4*len(y)//5)]
x_test, y_test = x[(4*len(x)//5)+1:], y[(4*len(y)//5)+1:]
'''


# Holmes specific
x_train, y_train = x[:84690], y[:84690]
x_test, y_test = x[:21150], y[:21150]


'''
# Copperfield specific
x_train, y_train = x[:290040], y[:290040]
x_test, y_test = x[290040:], y[290040:]
'''



# Model creation

# Old model not stateful
'''
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, 20, input_length = 1))
model.add(layers.Bidirectional(layers.LSTM(16, return_sequences=False, stateful=True),
                               input_shape=(len(hvec),20)))
model.add(layers.Dense(vocab_size, activation='softmax'))
'''

'''
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(input_dim = vocab_size, output_dim = 20, input_length = 1, batch_size = 32))
model.add(layers.Bidirectional(layers.LSTM(17, return_sequences=False, stateful=True, input_shape = (32, len(hvec), 20))))
model.add(layers.Dense(vocab_size, activation='softmax', input_shape = (32, 17)))
#print(model.summary())
'''

def buildModel(vocab_size, batch_size, in_length, embed_out, lstm_layers, lstm_out):
    if lstm_layers == 1:
        m = tf.keras.Sequential()
        m.add(tf.keras.layers.Embedding(input_dim = vocab_size, output_dim = embed_out, input_length = in_length, batch_size = batch_size))
        m.add(layers.Bidirectional(layers.LSTM(lstm_out, return_sequences=False, stateful=True)))
        m.add(layers.Dropout(0.5))
        m.add(layers.Dense(vocab_size, activation='softmax'))
        
    if lstm_layers == 2:
        m = tf.keras.Sequential()
        m.add(tf.keras.layers.Embedding(input_dim = vocab_size, output_dim = embed_out, input_length = in_length, batch_size = batch_size))
        m.add(layers.Bidirectional(layers.LSTM(lstm_out, return_sequences=True, stateful=True)))
        m.add(layers.Bidirectional(layers.LSTM(lstm_out, return_sequences=False, stateful=True)))
        m.add(layers.Dropout(0.5))
        m.add(layers.Dense(vocab_size, activation='softmax'))
        
    m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['loss', 'accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
    return m
    
model = buildModel(vocab_size = vocab_size, batch_size = 30, in_length = 1, embed_out = 16, lstm_layers = 1, lstm_out = 16)

model2 = buildModel(vocab_size = vocab_size, batch_size = 30, in_length = 1, embed_out = 16, lstm_layers = 2, lstm_out = 16)


# Set up checkpoints so that we can save model weights

# Directory where the checkpoints will be saved
#checkpoint_dir = r'C:\Users\penko\OneDrive\ML_Practice\Neural Net Practice\OG'
checkpoint_dir = r'C:\Users\penko\OneDrive\ML_Practice\Neural Net Practice\holmes05272020_1'
#checkpoint_dir = r'C:\Users\penko\OneDrive\ML_Practice\Neural Net Practice\copper05202020'

# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}-{loss:.4f}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True,
    monitor='loss',
    mode = 'min')

def trainModel(num_epochs, train_history = [], test_history = []):
    for i in range(num_epochs):
        print("---------------------------------------------------------------")
        print("- starting epoch {e}/{n} - ".format(e = i+1, n = num_epochs))
        train_history.append(model.fit(x_train, y_train, epochs=1, verbose=1, shuffle = False, callbacks=[checkpoint_callback]))
        model.reset_states()
        print("- TEST RESULTS -")
        test_history.append(model.evaluate(x_test, y_test))
        model.reset_states()
    return train_history, test_history
        
nepoch = 6
train_hist, test_hist = [], []
train_hist, test_hist = trainModel(10, train_hist, test_hist)


plt.plot([i+1 for i in range(len(test_hist))], [x.history['loss'][0] for x in train_hist], 'r',
         [i+1 for i in range(len(test_hist))], [x[0] for x in test_hist], 'b')


#model.save(r'C:\Users\penko\OneDrive\ML_Practice\Neural Net Practice\holmes_model_05202020.h5')
#model.save(r'C:\Users\penko\OneDrive\ML_Practice\Neural Net Practice\copperfield_model_05202020.h5')  
    
# Re-build model to take in batch size of 1 (instead of 30)

model = buildModel(vocab_size = vocab_size, batch_size = 1, in_length = 1, embed_out = 16, lstm_out = 16)

checkpoint_dir = r'C:\Users\penko\OneDrive\ML_Practice\Neural Net Practice\holmes05222020'
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

model.save(r'C:\Users\penko\OneDrive\ML_Practice\Neural Net Practice\holmes_model_05222020.h5')


# Predict word ('opium'... will this give us 'den'?)

in_text = ['watson']
#in_text = ['Holmes', 'I', 'want', 'to', 'tell', 'you']
enc = tokenizer.texts_to_sequences(in_text)
enc = np.array(enc)
yhat = model.predict_classes(enc, verbose=0, batch_size = 1)
yhat2 = model(enc)
yhat3 = model.predict(enc)
for pred in yhat:
    print(tokenizer.sequences_to_texts([[pred]]))
for word, index in tokenizer.word_index.items():
    if index == yhat:
        print(word)
        
        
temperature = 0.5
        

# currently takes in a list of strings as input. Could have chosen
# to take in one string and split it in the function. But for now I figure
# taking a list allows the user (me) to split it however I want
def generate_seq(model, tokenizer, seed_text, n_words, temp):
    if isinstance(seed_text, list):
        # flag that allows multiple "not in data set" warnings to be thrown
        invalidword = False
        # Loop through input words
        for s in seed_text:
            if not isinstance(s, str):
                print("All input words must be of type 'str'.")
                return
            if s not in tokenizer.word_index.keys():
                print("The word " + str(s) + " is not in the data set.")
                invalidword = True
        if invalidword:
            return
    else:   # if not list
        if not isinstance(seed_text, str):
            print("Input must be a string or list of strings.")
            return
        if seed_text not in tokenizer.word_index.keys():
            print("The word " + str(seed_text) + " is not in the data set.")
            return
    
    in_word, result = seed_text, seed_text
    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        e = tokenizer.texts_to_sequences([in_word])[0]
        e = np.array(e)
        # predict a word in the vocabulary
        #yhat = model.predict_classes(e, verbose=0, batch_size=1)
        if temp != 0:
            yh = model.predict(e, verbose=0, batch_size=1)
            yh = yh.squeeze()
            #print("temp type: " + str(type(temp)))
            yh = yh / temp
            #print("yh type after divide: " + str(type(yh)))
            
            #yhat_choice = np.array(tf.random.categorical(yh, num_samples=1)[-1,0])
            yhat_choice = np.random.choice(a=[x for x in range(vocab_size)], p=ss.softmax(yh))
        else:
            yhat_choice = model.predict_classes(e, verbose=0, batch_size=1)
            
        #print("yh type after categorical: " + str(type(yh)))
        #print("yhat_choice type: " + str(type(yhat_choice)))
        #print("yhat_choice :" + str(yhat_choice))
        #model.reset_states()
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat_choice:
                out_word = word
                break
        # append to input
        in_word, result = out_word, result + ' ' + out_word
    return result

# new idea 5/21/2020...
'''
I can acccomodate multiple words of input in the following way

If I give, say, five words of input, then I start by using the first word to
predict the second word. However, I dont store this prediction. I only use it
to get the first word into the state of the model. Then, I use the second word
to predict the third word, etc.
Eventually, I will use the final inputted word to predict the next (unknown)
word. From this point onward I will be storing the predictions as usual.
Because of the fact that I am relying on the model state, I will NOT
reset the state after each prediction. Only if I were to clear all of the
predictions/input and start over (like in the web app) would I reset the state.
'''
def generate_seq2(model, tokenizer, seed_text, n_words, top_k = 0, tmp = 1):
    if not isinstance(seed_text, list):
        seed_text = [seed_text]
    
    # flag that allows multiple "not in data set" warnings to be thrown
    invalidword = False
    # Loop through input words
    for s in seed_text:
        if not isinstance(s, str):
            print("All input words must be of type 'str'.")
            return
        if s not in tokenizer.word_index.keys():
            print("The word " + str(s) + " is not in the data set.")
            invalidword = True
    if invalidword:
        return
    
    print(seed_text)
    
    # Data cleaning, helping to match it with the data set
    for s in seed_text:
        s = s.lower().replace('\n',' ').replace('_','').replace('½','').replace("'", '').replace('"','').replace('‘', '').replace('’', '').replace('&c', 'etc').replace('“', '').replace('”', '').replace('â', 'a').replace('æ', 'ae').replace('è', 'e').replace('é', 'e').replace('œ', 'oe').replace('£', '').replace('à', 'a')
    
    seed_enc = np.array(tokenizer.texts_to_sequences(seed_text))
    
    # 'Generate' predicitons based on all input words except for the last one
    # This puts the words into the model state, and we dont save the predicitions
    for i in seed_enc[:-1]:
        model.predict(i, verbose = 0, batch_size = 1)
        
    # First input is the last element of seed_enc. Then we will set it to each
    # predicted word
    input_enc = seed_enc[-1:]
    choice_index_vec = []
    for _ in range(n_words):
        # Get softmax array of probabilities for each word
        print(input_enc)
        if tmp == 0:
            choice_index = model.predict_classes(input_enc, verbose = 0, batch_size = 1)[0]
            print("choice after prediction: " + str(choice_index))
            print("this choice equals the word: " + str(tokenizer.sequences_to_texts([[choice_index]])))
            choice_index_vec.append(choice_index)
            input_enc = [[choice_index]]
        else:
            yhatvec = model.predict(input_enc, verbose = 0, batch_size = 1)
            print("Type of yhatvec: " + str(type(yhatvec)))
            yhatvec = yhatvec.squeeze()
        
            # Only consider top k candidates
            if top_k != 0:
                # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array/23734295#23734295
                top_k_indices = np.argpartition(yhatvec, (-1*top_k))[(-1*top_k):]
                # Set all elements outside of top-k to zero
                for i in range(len(yhatvec)):
                    if i not in top_k_indices:
                        yhatvec[i] = 0
                
            yhatvec = yhatvec / tmp
            
            # Divide the resulting vector by its sum so that the probabilities
            # add to 1 again.
            yhatvec = yhatvec / sum(yhatvec)
            
            # Make a 'decision' on which word to predict
            # Add the index of the choice to vector
            choice_index = np.random.choice(a=[x for x in range(len(yhatvec))], p = yhatvec)
            
            #for word, index in tokenizer.word_index.items():
            #    if index == choice_index:
            #        out_word = word
            #        break
            
            choice_index_vec.append(choice_index)
            
            # Change input to previous prediction
            input_enc = [[choice_index]]
      
    print(choice_index_vec)
    word_choices = seed_text + tokenizer.sequences_to_texts([[x] for x in choice_index_vec])
    
    return word_choices
    

model.reset_states()
generate_seq(model, tokenizer, 'watson', 5, 0.002)
generate_seq(model, tokenizer, 'went', 5, 0.002)
generate_seq2(model, tokenizer, ['brother'], 5, 0, 0.000)
generate_seq2(model, tokenizer, ['holmes', 'brother'], 5, 0, 0.000)
generate_seq2(model, tokenizer, ['holmes', 'played', 'a'], 4, 8, 0.4)

for word, index in tokenizer.word_index.items():
    if index == 7893:
        print(index)
        print(word)