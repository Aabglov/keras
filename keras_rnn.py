# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
import os
import numpy as np
import random
import os
import word_helpers
import pickle
import time
import caffeine
import dialog_parser

from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.preprocessing import sequence
import numpy as np
# fix random seed for reproducibility
np.random.seed(36)


# PATHS -- absolute
SAVE_DIR = "conv"
CHECKPOINT_NAME = "conv_steps.ckpt"
PICKLE_PATH = "conv_tokenized.pkl"
DEBUG = False
dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(dir_path,"saved",SAVE_DIR,CHECKPOINT_NAME)
checkpoint_path = os.path.join(dir_path,"saved",SAVE_DIR)

GO =    dialog_parser.GO
UNK =   dialog_parser.UNK
PAD =   dialog_parser.PAD
EOS =   dialog_parser.EOS
SPLIT = dialog_parser.SPLIT

def save(obj,name):
    with open(os.path.join(dir_path,name),"wb+") as f:
        pickle.dump(obj,f)

def load(name):
    with open(os.path.join(dir_path,name),"rb") as f:
        return pickle.load(f)


try:
    input_seq = load("inputs.pkl")
    target_seq = load("targets.pkl")
    convs = load("convs.pkl")
    vocab = load("vocab.pkl")
    print("Loaded prepared data...")

except Exception as e:
    print("FAILED TO LOAD:")
    print(e)

    input_seq,target_seq,convs,vocab = dialog_parser.parseDialog()

    save(input_seq,"inputs.pkl")
    save(target_seq,"targets.pkl")
    save(convs,"convs.pkl")
    save(vocab,"vocab.pkl")

vocab_lookup = {}
reverse_vocab_lookup = {}
for i in range(len(vocab)):
    word = vocab[i]
    vocab_lookup[word] = i
    reverse_vocab_lookup[i] = word

# Remove the null character (0)
# uh..
for i in range(len(input_seq)):
    k = input_seq[i]
    input_seq[i] = [x for x in k if x != 0]

for i in range(len(target_seq)):
    k = target_seq[i]
    target_seq[i] = [x for x in k if x != 0]

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
max_seq_len = 100
embedding_vecor_length = 1028#32
top_words = len(vocab)
split_index = int(len(input_seq) * 0.8)


# load the dataset but only keep the top n words, zero the rest
X_train = input_seq[:split_index]
X_test = input_seq[split_index:]
y_train = target_seq[:split_index]
y_test = target_seq[split_index:]
# truncate and pad input sequences
X_train = sequence.pad_sequences(X_train, maxlen=max_seq_len)
X_test = sequence.pad_sequences(X_test, maxlen=max_seq_len)



######################## MODEL #############################
# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
