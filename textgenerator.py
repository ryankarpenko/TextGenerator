# -*- coding: utf-8 -*-

# https://towardsdatascience.com/build-a-text-generator-web-app-in-under-50-lines-of-python-9b63d47edabb

import panel as pn
import numpy as np

pn.extension()

# Load model

from tensorflow.keras.models import load_model
model = load_model('./holmes_model_05222020.h5')

# Load tokenizer

import pickle
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to generate next word

def generate_seq(model, tokenizer, seed_text, n_words, top_k = 0, tmp = 1):    
    # Slider wouldn't go down to the 0.001 level, so I put it to 0-100
    tmp = tmp / 100
    
    if not isinstance(seed_text, list):
        seed_text = [seed_text]
    
    # flag that allows multiple "not in data set" warnings to be thrown
    invalidword = False
    invalidwords = []
    # Loop through input words
    for s in seed_text:
        if not isinstance(s, str):
            raise KeyError('Input must be strings.')
        if s.lower() not in tokenizer.word_index.keys():
            invalidword = True
            invalidwords.append(s)
    if invalidword:
        if len(invalidwords) == 1:
            raise KeyError('The word ' + invalidwords[0] + ' is not in the data set.')
        else:
            raise KeyError('The words ' + ', '.join(invalidwords) + ' are not in the data set.')
        
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
        if tmp == 0:
            choice_index = model.predict_classes(input_enc, verbose = 0, batch_size = 1)[0]
            choice_index_vec.append(choice_index)
            input_enc = [[choice_index]]
        else:
            yhatvec = model.predict(input_enc, verbose = 0, batch_size = 1)
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
            
            choice_index_vec.append(choice_index)
            
            # Change input to previous prediction
            input_enc = [[choice_index]]
      
    word_choices = tokenizer.sequences_to_texts([[x] for x in choice_index_vec])
    
    return word_choices

# Create and link text input and generate text boxes 

text_input = pn.widgets.TextInput(name = 'Input Text:', width = 320)

generated_text = pn.pane.Markdown(object=text_input.value, width = 320)

generated_text_error = pn.pane.Markdown('', align = 'center', width = 500)

text_input.link(generated_text, value='object')

# Create functionality and widgets for Top K Candidates parameter

def top_k_input_to_slider(target, event):
    if (isinstance(event.new, int) or isinstance(event.new, float)) and (0 <= event.new <= 1000):
            target.value = event.new
            top_k_error.object = ''
    else:
        top_k_input.value = event.old
        top_k_error.object = 'K value must be a number between 0 and 1000.'
        
def top_k_slider_to_input(target, event):
    top_k_input.value = top_k_slider.value


top_k_slider = pn.widgets.IntSlider(name='Top K Candidates', show_value = False, start = 0, end = 1000, step = 1, value = 0, width = 200)
top_k_input = pn.widgets.LiteralInput(name = '', value = int(top_k_slider.value), type = int, width = 100, align = 'end')
top_k_error = pn.pane.Markdown('', align = 'center', width = 500)

top_k_input.link(top_k_slider, callbacks = {'value': top_k_input_to_slider})
top_k_slider.link(top_k_input, callbacks = {'value': top_k_slider_to_input})
top_k_input.value = top_k_slider.value

top_k_row = pn.Row(top_k_slider, top_k_input, top_k_error)


# Create functionality and widgets for Temperature parameter

def temp_input_to_slider(target, event):
    if (isinstance(event.new, int) or isinstance(event.new, float)) and (0 <= event.new <= 100):
            target.value = event.new
            temperature_error.object = ''
    else:
        temperature_input.value = event.old
        temperature_error.object = 'Temperature value must be a number between 0 and 100.'
        
def temp_slider_to_input(target, event):
    if (isinstance(event.new, int) or isinstance(event.new, float)) and (0 <= event.new <= 100):
        target.value = event.new


temperature_slider = pn.widgets.FloatSlider(name='Temperature', show_value = False, start=0, end=100, step=0.01, value=0.0, width = 200)
temperature_input = pn.widgets.LiteralInput(name = '', value = float(temperature_slider.value), type = float, width = 100, align = 'end')
temperature_error = pn.pane.Markdown('', align = 'center', width = 500)

temperature_input.link(temperature_slider, callbacks = {'value': temp_input_to_slider})
temperature_slider.link(temperature_input, callbacks = {'value': temp_slider_to_input})
temperature_input.value = temperature_slider.value

temperature_row = pn.Row(temperature_slider, temperature_input, temperature_error, sizing_mode = 'stretch_width')

# Create functionality for "Generate" button

def click_generate(event):
    temp_object = generated_text.object
    
    object_formatted = ''.join([temp_object[i] for i in range(len(temp_object)) if (temp_object[i].isalnum() or temp_object[i] == ' ')])
    
    object_segmented = object_formatted.split()
    
    if object_segmented != []:
        try:
            pred = generate_seq(model, tokenizer, seed_text = object_segmented, n_words = 1, 
                        top_k = top_k_slider.value, tmp = temperature_slider.value)
            generated_text_error.object = ''
        except KeyError as err:
            print("Error was thrown. Error value: " + str(err)[1:-1])
            print("Type of err: " + str(type(err)))
            generated_text_error.object = str(err)[1:-1]
            
        generated_text.object = generated_text.object + " " + str(pred[0])
    else:
        generated_text.object = text_input.value


# Create functionality for "Clear" button

clear_button = pn.widgets.Button(name = 'Clear', button_type = 'primary', align = 'end', width = 40)

def clear_input(event):
    text_input.value = ''
    generated_text_error.object = ''
    
clear_button.on_click(clear_input)

generate_button = pn.widgets.Button(name="Generate",button_type="warning", width = 200, margin = (10,0,0,10))
generate_button.on_click(click_generate)

text_input_row = pn.Row(text_input, clear_button)
generate_button_row = pn.Row(generate_button, generated_text_error)

# Title, scroller and description for app display

title = pn.pane.Markdown('# <span style="font-family:Rockwell;font-size:1.25em;">**Sherlock Holmes Text Generator**</span>', width = 600)
description = pn.pane.Markdown("""This neural network was given the first book of Sherlock Holmes short stories as data.
From that text, it learned how to predict the next word in a sentence, given the words that came before it.
In order to generate a new word, it takes a word or phrase as input from the user.
Then it calculates the probability that each word will come next, for all the words in its vocabulary.
Finally, it randomly selects one of the possible words based on their weighted probability.

The generated results can vary greatly depending on a few important variables:

**Top K:** Setting a value of K means the next generated word will be selected from only the K highest-probability words.
For example, if K = 5, the network will only choose from the 5 most likely words.
Set K to 0 to consider all words without restriction.

**Temperature:** Temperature impacts how heavily the network favors the most likely words.
Lower temperature values will feel more reasonable, but the results will be less diverse.
Higher temperature values will feel more random, but they might not make as much sense.
Try to find the sweet spot that seems most realistic to you!

""", width = 1000)

scroll = pn.pane.HTML("<marquee scrollamount='10'><b>Elementary, my dear Watson! Using the powers of deduction, this neural network will take a given sentence and predict the next word. Just adjust the parameters, type in a word or sentence, and generate as many new words as you please!</b></marquee>",
                   width = 1000)

# App object

final_app = pn.Column(title, scroll, description, top_k_row, temperature_row, text_input_row, generate_button_row, generated_text, margin = (10, 10, 10, 10))

final_app.servable()

final_app