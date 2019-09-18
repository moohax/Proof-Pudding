# from keras.models import Sequential
# from keras.models import load_model

# from core import models
# from core import preprocessing

# class ProofPudding(object):
#     def __init__(self, vocab_size, hidden_size, input_length, classifier, debug=True):
#         self.debug = debug
#         self.model = Sequential()
#         self.model.add(LSTM(128, input_shape=(maxlen, len(chars))))
#         self.model.add(Dense(len(chars), activation='softmax'))
#         self.optimizer = RMSprop(learning_rate=0.01)
#         self.model.compile(loss=CopyCatLoss, optimizer=self.optimizer)
#         self.classifier = load_model(classifier)


#     def custom_loss(i):
#         def copy_cat_loss(y_true, )

#     return keras.losses.binary_crossentropy(y_t,y_p)

#     def CopyCatLoss(y_true, y_pred):
#         x = self.classifier
#         y_pred = self.classifier.predict(self.model)
#         y_true = 999
#         loss = y_true - y_pred
#         return loss
      
#     def sample(preds, temperature=1.0):
#         # Functions from keras-team/keras/blob/master/examples/lstm_text_generation.py
#         # helper function to sample an index from a probability array
#         preds = np.asarray(preds).astype('float64')
#         preds = np.log(preds)/temperature
#         exp_preds = np.exp(preds)
#         preds = exp_preds/np.sum(exp_preds)
#         probas = np.random.multinomial(1, preds, 1)
#         return np.argmax(probas)

#     def generate_next(text, num_generated=1):
#         word_idxs = [word2idx(word) for word in text.lower().split()]
#         for i in range(num_generated):
#             prediction = model.predict(x=np.array(word_idxs))
#             idx = sample(prediction[-1], temperature=0.7)
#             word_idxs.append(idx)
        
#         return ' '.join(idx2word(idx) for idx in word_idxs)


#     def on_epoch_end(epoch, _):
#         # Function invoked at end of each epoch. Prints generated text.
#         print()
#         print('----- Generating text after Epoch: %d' % epoch)

#         start_index = random.randint(0, len(text) - maxlen - 1)
#         for diversity in [0.2, 0.5, 1.0, 1.2]:
#             print('----- diversity:', diversity)

#             generated = ''
#             sentence = text[start_index: start_index + maxlen]
#             generated += sentence
#             print('----- Generating with seed: "' + sentence + '"')
#             sys.stdout.write(generated)

#             for i in range(400):
#                 x_pred = np.zeros((1, maxlen, len(chars)))
#                 for t, char in enumerate(sentence):
#                     x_pred[0, t, char_indices[char]] = 1.

#                 preds = model.predict(x_pred, verbose=0)[0]
#                 next_index = sample(preds, diversity)
#                 next_char = indices_char[next_index]

#                 sentence = sentence[1:] + next_char

#                 sys.stdout.write(next_char)
#                 sys.stdout.flush()
#             print()

#         print_callback = LambdaCallback(on_epoch_end=on_epoch_end)


#         # create ngrams
#         total_words = len(tokenizer.word_index) + 1
        
#         input_sequences = []
#         for i in range(1, len(text_sequences)):
#             n_gram_sequence = text_sequences[:i+1]
#             input_sequences.append(n_gram_sequence)

#         input_sequences = pad_sequences(input_sequences, 300 , padding='pre')
#         text_matrix, score_matrix = input_sequences[:,:-1],input_sequences[:,-1]
#         score_matrix = ku.to_categorical(label, num_classes=total_words)



# ProofPudding = ProofPudding()