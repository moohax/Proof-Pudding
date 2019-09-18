from alive_progress import alive_bar

#import numpy
from numpy import array

#import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import LSTM
import keras.backend as K

# 16fp training
# Default is 1e-7 which is too small for float16.
# Without adjusting the epsilon, we will get NaN 
# predictions because of divide by zero problems
K.set_epsilon(1e-4) 
K.set_floatx('float16')

# copy cat model
def create_neural_network(input_dim, hidden_dim=64, activation='sigmoid'):
    model = Sequential()
    model.add(Dense(hidden_dim, activation=activation, input_dim=input_dim))
    model.add(Dense(hidden_dim, activation=activation))
    model.add(Dense(1, activation=activation))
    model.compile(loss='mae', optimizer='adam', metrics=['mae'])

    return model

def get_insights(model, inputs, tokenizer):
    
    insights = {}

    with alive_bar(len(inputs), bar='blocks') as bar:
        for sample in inputs:
            bar()
            base_prediction = model.predict(numpy.array([sample]))[0][0]

            for i,word_is_set in enumerate(sample):
                if i is 0 or not word_is_set: continue # first index is reserved
                
                word = tokenizer.index_word[i]
                alt_sample = numpy.copy(sample)
                alt_sample[i] = 0
                new_prediction = model.predict(numpy.array([alt_sample]))[0][0]            

                if word not in insights:
                    insights[word] = [0, 0]

                insights[word][0] += 1
                insights[word][1] += (base_prediction - new_prediction)

    insights = dict([(k, i[1] / i[0]) for k,i in insights.items()])

    return sorted(insights.items(), key=lambda x: x[1], reverse=True) 

def create_neural_network_lstm(vocab_size, hidden_size, input_length, activation='sigmoid'):
    model = Sequential()
    model.add(Embedding(vocab_size, hidden_size))
    model.add(LSTM(hidden_size))
    model.add(Dense(1, activation=activation))
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])

    return model


# Copy cat model lstm 
# class LSTM(object):
#     def __init__(self, vocab_size, hidden_size, input_length, debug=True):
#         self.debug = debug
#         self.model = Sequential()
#         self.model.add(Embedding(vocab_size, hidden_size))
#         self.model.add(keras.layers.LSTM(hidden_size))
#         self.model.add(Dense(1, activation='sigmoid')) 
#         self.model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae']) 

#         if self.debug:
#             print(self.model.summary())

#     def train(self, x, y, epochs, batch_size, validation_split = 0):
#         tb_callback = keras.callbacks.TensorBoard(log_dir='core\\log\\', histogram_freq=0, write_graph=True, write_images=True)
#         early_stop = EarlyStopping(monitor='val_mean_absolute_error', mode='min', verbose=1)
#         history = self.model.fit(x, y, nb_epoch=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[tb_callback, early_stop])
#         return history