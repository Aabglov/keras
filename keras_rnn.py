# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
import os
import numpy as np
import random
import os
import word_helpers
import pickle
import time
#import caffeine
import dialog_parser

from random import shuffle

from keras.models import Model,load_model
from keras.layers import Input, LSTM, Dense, Embedding
from keras.preprocessing import sequence
from keras.utils import to_categorical
import numpy as np
# fix random seed for reproducibility
np.random.seed(36)


# PATHS -- absolute
SAVE_DIR = "conv"
CHECKPOINT_NAME = "conv_steps.ckpt"
PICKLE_PATH = "conv_tokenized.pkl"
FINAL_SAVE_PATH = 'saved/rnn/s2s_final.h5'
DEBUG = False
dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(dir_path,"saved",SAVE_DIR,CHECKPOINT_NAME)
checkpoint_path = os.path.join(dir_path,"saved",SAVE_DIR)

GO =    dialog_parser.GO
UNK =   dialog_parser.UNK
PAD =   dialog_parser.PAD
EOS =   dialog_parser.EOS
SPLIT = dialog_parser.SPLIT

def save(obj,name,protocol=False):
    if protocol:
        with open(os.path.join(dir_path,name),"wb+") as f:
            pickle.dump(obj,f,protocol=protocol)
    else:
        with open(os.path.join(dir_path,name),"wb+") as f:
            pickle.dump(obj,f)

def load(name):
    with open(os.path.join(dir_path,name),"rb") as f:
        return pickle.load(f)


class Batcher:
    def __init__(self, encoder_input_data, decoder_input_data, decoder_target_data, batch_size):
        self.encoder_input =  encoder_input_data
        self.decoder_input =  decoder_input_data
        self.decoder_target = decoder_target_data
        self.index = 0
        self.batch_size = batch_size
        # This should be the same as using
        # decoder_input and decoder_target because
        # They're all the same length
        self.length = len(self.encoder_input)

    def next(self):
        begin = self.index
        # If the batch would go past the end of
        # our dataset we instead return
        # the remaining entries only.
        # Prevents overflow
        if (self.index + self.batch_size) > self.length:
            end = -1
        else:
            end = self.index + self.batch_size
        output = [self.encoder_input[begin:end],
                  self.decoder_input[begin:end],
                  to_categorical(self.decoder_target[begin:end],num_classes=vocab_len).reshape((-1,max_seq_len,vocab_len))]

        self.index += 1
        self.index = self.index % self.length
        return output

try:
    input_seq = load("inputs.pkl")
    target_seq = load("targets.pkl")
    convs = load("convs.pkl")
    vocab = load("vocab.pkl")
    print("Loaded prepared data...")

    # save(input_seq,"inputs.pkl",protocol=2)
    # save(target_seq,"targets.pkl",protocol=2)
    # save(convs,"convs.pkl",protocol=2)
    # save(vocab,"vocab.pkl",protocol=2)


except Exception as e:
    print("FAILED TO LOAD:")
    print(e)

    input_seq,target_seq,convs,vocab = dialog_parser.parseDialog()

    save(input_seq,"inputs.pkl")
    save(target_seq,"targets.pkl")
    save(convs,"convs.pkl")
    save(vocab,"vocab.pkl")



######################## DATA PREP ########################
vocab_lookup = {}
reverse_vocab_lookup = {}
for i in range(len(vocab)):
    word = vocab[i]
    vocab_lookup[word] = i
    reverse_vocab_lookup[i] = word

target_input_seq = []
target_output_seq = []
for i in range(len(target_seq)):
    k = target_seq[i]
    target_seq[i] = [x for x in k if x != 0]
    target_input_seq.append([vocab_lookup[GO]] + target_seq[i])
    target_output_seq.append(target_seq[i] + [vocab_lookup[EOS]])

if DEBUG:
    print(input_seq[1])
    print(" ".join([reverse_vocab_lookup[i] for i in input_seq[1]]))
    print(" ".join([reverse_vocab_lookup[i] for i in target_seq[0]]))
    print(convs[0].lines[1])


batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
# max length of 100 excludes about 1200 of 300k samples.
# max length of 200 excludes about 100 of 300k
max_seq_len = 30
vocab_len = len(vocab)
save_interval = 100
log_interval = 10

print("vocab length: ",vocab_len)

embedding_dim = 128

# truncate and pad input sequences
pad_val = vocab_lookup[PAD]
padded_input = sequence.pad_sequences(input_seq, maxlen=max_seq_len,value=pad_val)
padded_target_input = sequence.pad_sequences(target_input_seq, maxlen=max_seq_len,value=pad_val)
padded_target_output = sequence.pad_sequences(target_output_seq, maxlen=max_seq_len,value=pad_val)

# Turn our sequences into arrays
encoder_input_data = np.asarray(padded_input,dtype='float32')
decoder_input_data = np.asarray(padded_target_input ,dtype='float32')
decoder_target_data = np.asarray(padded_target_output,dtype='float32')

print(decoder_input_data.shape)
print(decoder_target_data.shape)
decoder_target_hold = []
for d in decoder_target_data:
    decoder_target_hold.append(to_categorical(d,num_classes=vocab_len))
decoder_target_data = np.asarray(decoder_target_hold)
print(decoder_target_data)


# decoder_target = np.asarray(padded_target_output,dtype='float32')
# decoder_target_data = np.zeros((decoder_target.shape[0],decoder_target.shape[1],vocab_len))
# for i in range(decoder_target.shape[0]):
#     for j in range(decoder_target.shape[1]):
#         k = int(decoder_target[i,j])
#         decoder_target_data[i,j,k] = 1



######################## MODEL #############################
try:
    model = load_model(FINAL_SAVE_PATH)
    print("Loaded saved model")
except:
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(max_seq_len,))
    print("No saved model, creating...")

    x = Embedding(vocab_len, embedding_dim,input_length=max_seq_len)(encoder_inputs)
    x, state_h, state_c = LSTM(latent_dim,return_sequences=True,return_state=True)(x)
    x, state_h, state_c = LSTM(latent_dim,return_sequences=True,return_state=True)(x)
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(max_seq_len,))
    x = Embedding(vocab_len, embedding_dim,input_length=max_seq_len)(decoder_inputs)
    x = LSTM(latent_dim, return_sequences=True)(x, initial_state=encoder_states)
    x = LSTM(latent_dim, return_sequences=True)(x)
    decoder_outputs = Dense(vocab_len, activation='softmax')(x)
    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Compile & run training
    model.compile(optimizer='adam', loss='categorical_crossentropy')

# Note that `decoder_target_data` needs to be one-hot encoded,
# rather than sequences of integers like `decoder_input_data`!
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
         batch_size=batch_size,
         epochs=epochs,
         validation_split=0.2)

# batcher = Batcher(encoder_input_data, decoder_input_data, decoder_target_data, batch_size)
#
# # Define helper functions for printing
# # conversations for training evaluation
# def printEncoderInput(encoder_input):
#     enc = encoder_input[0]
#     enc_words = [reverse_vocab_lookup[w] for w in enc]
#     print("INPUT: " + " ".join(enc_words))
#
# def printPred(pred_vec,label="PRED"):
#     pred = pred_vec[0]
#     pred_words = [np.random.choice(vocab,size=1,p=w)[0] for w in pred]
#     print(label + ": " + " ".join(pred_words))
#
# num_batches = batcher.length // batcher.batch_size
#
# try:
#     for epoch in range(epochs):
#         # Iterate through our dataset
#         for _ in range(num_batches):
#             encoder_input_batch,decoder_input_batch,decoder_target_batch = batcher.next()
#
#             # Train the generator
#             loss = model.train_on_batch([encoder_input_batch, decoder_input_batch], decoder_target_batch)
#
#             # Plot the progress
#             iteration = (epoch*num_batches) + _
#             print("%d loss: %f" % (iteration, loss))
#
#             if _ % log_interval == 0:
#                 pred = model.predict([encoder_input_batch, decoder_input_batch])
#                 printEncoderInput(encoder_input_batch)
#                 printPred(pred)
#                 printPred(decoder_target_batch,label="TRUE")
#
#             # If at save interval => save generated image samples
#             if _ % save_interval == 0:
#                 model.save('saved/rnn/s2s_{}.h5'.format(iteration))
#
#
# except KeyboardInterrupt:
#     print("Early exit")



# Save model
model.save(FINAL_SAVE_PATH)

HODOR

# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


for seq_index in range(100):
    # Take one sequence (part of the training test)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)
