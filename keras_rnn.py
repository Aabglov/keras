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
import numpy as np
# fix random seed for reproducibility
np.random.seed(36)


# PATHS -- absolute
SAVE_DIR = "conv"
CHECKPOINT_NAME = "conv_steps.ckpt"
PICKLE_PATH = "conv_tokenized.pkl"

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
print(input_seq[1])
print(" ".join([reverse_vocab_lookup[i] for i in input_seq[1]]))
print(" ".join([reverse_vocab_lookup[i] for i in target_seq[0]]))
print(convs[0].lines[0])
print(convs[0].lines[1])

HODOR

batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.

# Vectorize the data.
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
lines = open(data_path).read().split('\n')
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text = line.split('\t')
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)




max_seq_len = 100


# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# truncate and pad input sequences
X_train = sequence.pad_sequences(X_train, maxlen=max_seq_len)
X_test = sequence.pad_sequences(X_test, maxlen=max_seq_len)

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
