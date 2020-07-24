# TextGenerator
Neural network text generator trained on the stories of Sherlock Holmes.

I trained this LSTM neural network using TensorFlow in Python, with word embeddings and softmax activation to obtain a probability for each word in the vocabularity. The UI was implemented as a Panel web app, where you can adjust the parameters for the model prediction and try out the network. You can make your input an individual word or even a complete sentence! Just remember that some common words don't show up in the original dataset and therefore can't be used as input. Depending on the parameters and input you choose, the results can either be silly or surprisingly realistic!

Note: The webpage might take a minute to load, since Heroku needs to "wake up" the server if the page hasn't been viewed in a while.

Data was obtained from Project Gutenberg: https://www.gutenberg.org/
