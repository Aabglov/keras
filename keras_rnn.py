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
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
np.random.seed(36)


# PATHS -- absolute
SAVE_DIR = "dict" # mtg
CHECKPOINT_NAME = "dict_steps.ckpt" # "mtg_rec_char_steps.ckpt"
DATA_NAME = "dictionary.json" # "cards_tokenized.txt"
PICKLE_PATH = "dict_tokenized.pkl" #"mtg_tokenized_wh.pkl"
SUBDIR_NAME = "dictionary"

dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(dir_path,"saved",SAVE_DIR,CHECKPOINT_NAME)
checkpoint_path = os.path.join(dir_path,"saved",SAVE_DIR)
data_path = os.path.join(dir_path,"data",SUBDIR_NAME,DATA_NAME)


try:
    with open(os.path.join(dir_path,"data",PICKLE_PATH),"rb") as f:
        WH = pickle.load(f)
    print("word_helpers object found")
except Exception as e:
    print(e)
    # What's with those weird symbols?
    # u'\xbb' is our GO symbol (»)
    # u'\xac' is our UNKNOWN symbol (¬)
    # u'\xa4' is our END symbol (¤)
    # They're arbitrarily chosen, but
    # I think they both:
    #   1). Are unlikely to appear in regular data, let alone cleaned data.
    #   2). Look awesome.
    vocab = [u'\xbb','|', '5', 'c', 'r', 'e', 'a', 't', 'u', '4', '6', 'h', 'm', 'n', ' ', 'o', 'd', 'l', 'i', '7', \
             '8', '&', '^', '/', '9', '{', 'W', '}', ',', 'T', ':', 's', 'y', 'b', 'f', 'v', 'p', '.', '3', \
             '0', 'A', '1', 'w', 'g', '\\', 'E', '@', '+', 'R', 'C', 'x', 'B', 'G', 'O', 'k', '"', 'N', 'U', \
             "'", 'q', 'z', '-', 'Y', 'X', '*', '%', '[', '=', ']', '~', 'j', 'Q', 'L', 'S', 'P', '2',u'\xac',u'\xf8',u'\xa4',u'\u00BB']

    # Load mtg tokenized data
    # Special thanks to mtgencode: https://github.com/billzorn/mtgencode
    # with open(data_path,"r") as f:
    #     # Each card occupies its own line in this tokenized version
    #     raw_txt = f.read()#.split("\n")
    # WH = word_helpers.WordHelper(raw_txt, vocab)
    WH = word_helpers.JSONHelper(data_path,vocab)
    print("word helpers object NOT found, re-created")

    # Save our WordHelper
    with open(os.path.join(dir_path,"data",PICKLE_PATH),"wb") as f:
        pickle.dump(WH,f)
    print("new word helpers object saved")


args = {
    'learning_rate':3e-4,
    'grad_clip':5.0,
    'n_input':WH.vocab.vocab_size,
    'n_classes':WH.vocab.vocab_size,
    'lstm_size':512,
    'num_layers':3, #2
    'num_steps':50 #250
}


# Network Parameters
LEARNING_RATE = args['learning_rate']
GRAD_CLIP = args['grad_clip']
N_INPUT = args['n_input']
N_CLASSES = args['n_classes']
LSTM_SIZE = args['lstm_size']
NUM_LAYERS = args['num_layers']
NUM_STEPS = args['num_steps']


# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# truncate and pad input sequences
max_review_length = 50
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

HODOR
# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, nb_epoch=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
