# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
import os
import numpy as np
import random
import sys
import pickle
import time
#import caffeine
from random import shuffle

from keras.models import Model,load_model
from keras.layers import Input, LSTM, Dense, Embedding
from keras import backend as K
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
import numpy as np
from tensorflow import one_hot
# fix random seed for reproducibility
np.random.seed(36)


# PATHS -- absolute
SAVE_DIR = "rap"
CHECKPOINT_NAME = "rap_char_steps.ckpt"
DATA_NAME = "ohhla.txt"
PICKLE_PATH = "rap_rh.pkl"
SUBDIR_NAME = "rap"
FINAL_SAVE_PATH = 'saved/rnn/s2s_final.h5'
DEBUG = False
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH = os.path.join(DIR_PATH,"saved",SAVE_DIR,CHECKPOINT_NAME)
CHECKPOINT_PATH = os.path.join(DIR_PATH,"saved",SAVE_DIR)

# Retrieve the helper functions in the other repo 
TENSORFLOW_PATH = DIR_PATH.replace("keras","tensorflow")
sys.path.insert(0, TENSORFLOW_PATH)
from helpers import rap_helper,rap_parser


def save(obj,name,protocol=False):
    if protocol:
        with open(os.path.join(DIR_PATH,name),"wb+") as f:
            pickle.dump(obj,f,protocol=protocol)
    else:
        with open(os.path.join(DIR_PATH,name),"wb+") as f:
            pickle.dump(obj,f)

def load(name):
    with open(os.path.join(DIR_PATH,name),"rb") as f:
        return pickle.load(f)


DIR_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH = os.path.join(DIR_PATH,"saved",SAVE_DIR,CHECKPOINT_NAME)
CHECKPOINT_PATH = os.path.join(DIR_PATH,"saved",SAVE_DIR)
data_path = os.path.join(DIR_PATH,"data",SUBDIR_NAME,DATA_NAME)

MAX_SEQ_LEN = 100

try:
    with open(os.path.join(CHECKPOINT_PATH,PICKLE_PATH),"rb") as f:
        RH = pickle.load(f)
except Exception as e:
    print(e)
    songs,parsed_vocab = rap_parser.getSongs()

    RH = rap_helper.SongBatcher(songs,parsed_vocab)

    # Length investigation
    # Looks like there are only about 1000 songs (out of 19k)
    # that contain batches 100 words long or longer
    REMOVE_BIG_SEQ = True
    if REMOVE_BIG_SEQ:
        len_dict = {}
        songs_to_remove = []
        for i in range(len(RH.songs)):
            s = RH.songs[i]
            for line in s:
                l = len(line)
                if l not in len_dict:
                    len_dict[l] = [line]
                else:
                    len_dict[l].append(line)
                if l > MAX_SEQ_LEN:
                    songs_to_remove.append(i)
        k = list(len_dict.keys())
        k.sort()
        # total = 0
        # for i in [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 127, 128, 129, 130, 131, 133, 134, 136, 140, 142, 143, 144, 146, 147, 148, 150, 151, 152, 153, 154, 161, 162, 163, 170, 171, 172, 173, 180, 187, 198, 202, 203, 209, 213, 215, 238, 245, 249, 264, 312, 316, 565]:
        #     total += len(len_dict[i])
        # print("TOTAL: {}".format(total))
        songs_to_remove = list(set(songs_to_remove))
        print("Number of songs to remove: {}".format(len(songs_to_remove)))
        print("Number of songs before removal: {}".format(len(RH.songs)))
        songs_to_remove.sort()
        for i in songs_to_remove[::-1]:
            del RH.songs[i]
        print("Number of songs after removal: {}".format(len(RH.songs)))

    print("parsed vocab length: {}".format(len(parsed_vocab)))

    # Save our Rap Helper
    with open(os.path.join(CHECKPOINT_PATH,PICKLE_PATH),"wb") as f:
        pickle.dump(RH,f)




vocab_len = len(vocab)
batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
# max length of 100 excludes about 1200 of 300k samples.
# max length of 200 excludes about 100 of 300k
max_seq_len = 30
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

def hack_loss(y_true_uncat):
    def custom_loss(y_true, y_pred):
        #y_true_cat = to_categorical(y_true,num_classes=vocab_len)
        y_true_cat = one_hot(y_true_uncat,depth=vocab_len)
        return categorical_crossentropy(y_true_cat,y_pred)
        #return nn.sparse_softmax_cross_entropy_with_logits(labels=y_true_uncat,logits=y_pred)
    return custom_loss

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

    decoder_target_input = Input(shape=(max_seq_len,),dtype='int32')

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(max_seq_len,))
    x = Embedding(vocab_len, embedding_dim,input_length=max_seq_len)(decoder_inputs)
    x = LSTM(latent_dim, return_sequences=True)(x, initial_state=encoder_states)
    x = LSTM(latent_dim, return_sequences=True)(x)
    decoder_outputs = Dense(vocab_len, activation="softmax")(x)
    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs, decoder_target_input], decoder_outputs)



    # Compile & run training
    model.compile(optimizer='adam', loss=hack_loss(decoder_target_input))

# Note that `decoder_target_data` needs to be one-hot encoded,
# rather than sequences of integers like `decoder_input_data`!
num_samples = encoder_input_data.shape[0]
model.fit([encoder_input_data, decoder_input_data, decoder_target_data],
         np.zeros((num_samples,max_seq_len,1)),
         batch_size=batch_size,
         epochs=epochs,
         validation_split=0.2)

# batcher = Batcher(encoder_input_data, decoder_input_data, decoder_target_data, batch_size)
#
# # Define helper functions for printing
# # conversations for training evaluation
# def printEncoderInput(encoder_input,label="INPUT"):
#     enc = encoder_input[0]
#     enc_words = [reverse_vocab_lookup[w] for w in enc]
#     print(label + ": ".join(enc_words))
#
# def printPred(pred_vec,label="PRED"):
#     pred = pred_vec[0]
#     pred_words = []
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
#             unused_output = np.zeros((batch_size,max_seq_len,vocab_len))
#             # Train the generator
#             loss = model.train_on_batch([encoder_input_batch, decoder_input_batch, decoder_target_batch], unused_output)
#
#             # Plot the progress
#             iteration = (epoch*num_batches) + _
#             print("%d loss: %f" % (iteration, loss))
#
#             if _ % log_interval == 0:
#                 unused_pred = np.zeros(decoder_target_batch.shape)
#                 pred = model.predict([encoder_input_batch, decoder_input_batch, unused_pred])
#                 printEncoderInput(encoder_input_batch)
#                 printPred(pred)
#                 printEncoderInput(decoder_target_batch,"TRUE")
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
