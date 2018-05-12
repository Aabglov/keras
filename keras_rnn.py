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


class Seq2Seq:
    def __init__(self,RH):
        # TRAINING PARAMETERS
        self.LEARNING_RATE = 3e-4
        self.GRAD_CLIP = 5.0
        self.LSTM_SIZE = 512
        self.NUM_LAYERS = 3
        # Since songs can have differing numbers of lines
        # it doesn't make sense to have multiple songs represented in a batch.
        # At least not yet.
        self.BATCH_SIZE = 1
        self.NUM_EPOCHS = 100
        self.NUM_STEPS = 100 # 50
        self.DISPLAY_STEP = 10#25
        self.SAVE_STEP = 1
        self.DECAY_RATE = 0.97
        self.DECAY_STEP = 5
        self.DROPOUT_KEEP_PROB = 0.8 #0.5
        self.TEMPERATURE = 1.0
        self.NUM_PRED = 50
        self.already_trained = 0
        self.N_CLASSES = 98
        self.vocab_obj = RH.vocab
        self.vocab = vocab_obj.vocab
        self.PRIME_TEXT = u"»"

        self.input_shape = (None,1)
        optimizer = Adam(0.0002, 0.5)

        # Build and compile the decoder
        try:
            self.decoder = load_model(DEC_MODEL_PATH)
        except OSError as e:
            print("decoder model not found, building...")
            self.decoder = self.build_decoder()
        self.decoder.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the encoder
        try:
            self.encoder = load_model(ENC_MODEL_PATH)
        except OSError as e:
            print("encoder model not found, building...")
            self.encoder = self.build_encoder()
        self.encoder.compile(loss='binary_crossentropy', optimizer=optimizer)

        # The encoder takes noise as input and generated imgs
        input = Input(shape=(self.BATCH_SIZE,1))
        encoder_output,state_h,state_c = self.encoder(z)

        # The valid takes generated images as input and determines validity
        decoder_output = self.decoder(encoder_output)

        # The combined model  (stacked encoder and decoder) takes
        # noise as input => generates images => determines validity
        self.combined = Model(input,decoder_output)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_encoder(self):

        in_shape = (self.BATCH_SIZE,1)
        in_batch = Input(shape=in_shape)

        model = Sequential()
        model.add(Embedding(self.N_CLASSES, self.LSTM_SIZE))
        model.add(LSTM(self.LSTM_SIZE, return_state=False))
        model.add(LSTM(self.LSTM_SIZE, return_state=False))
        model.add(LSTM(self.LSTM_SIZE, return_state=True))

        model.summary()

        encoder_output,state_h,state_c = model(in_batch)

        return Model(in_batch,encoder_output,state_h,state_c)

    def build_decoder(self):

        in_shape = (self.BATCH_SIZE,1)
        in_batch = Input(shape=in_shape)

        model = Sequential()
        model.add(LSTM(self.LSTM_SIZE, return_state=True))
        model.add(LSTM(self.LSTM_SIZE, return_state=True))
        model.add(LSTM(self.LSTM_SIZE, return_state=True))

        model.summary()

        encoder_output = model(in_batch)

        return Model(in_batch,encoder_output)

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
