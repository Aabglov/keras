# IMPORTS
import random
import os
import numpy as np # Used for One-hot encoding
import json # Used for json helper
import ast # USed to parse List from string ("['a','b',c']")

from nltk import word_tokenize # Word tokenizer
import re

EOS_SEARCH = "[\.,?!]+"
REGEX_SEARCH = '[^0-9a-zA-Z.,?!]+'
GO = u'\xbb'
UNK = u'\xac'
PAD = u'\xf8'
EOS = u'\xa4'
SPLIT = u'\u00BB'

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(DIR_PATH,"data","dialog")
STRUCTURE_PATH = os.path.join(DATA_PATH,"movie_conversations.txt")
LINES_PATH = os.path.join(DATA_PATH,"movie_lines.txt")
SAVE_PATH = os.path.join(DATA_PATH,"parsed")
SPACER = " +++$+++ " # Arbitrary spacer token used by dataset
#DEBUG = False
DEBUG = True

REMOVE_SINGLES = True

class Conversation:
    def __init__(self,subject1,subject2,lines):
        self.subject1 = subject1
        self.subject2 = subject2
        self.lines = lines

def preTokenize(text):
    text = text.replace("."," . ")
    text = text.replace(","," , ")
    text = text.replace("?"," ? ")
    text = text.replace("!"," ! ")
    # Separate numbers into their own entries for vocabulary.
    # We don't want to have every 3 digit number as its own word.
    for num in [0,1,2,3,4,5,6,7,8,9]:
        text = text.replace(str(num)," "+str(num)+" ")
    return text

def parseDialog():
    convs = []
    lines = {}
    vocabulary = [GO ,UNK ,PAD ,EOS ,SPLIT]
    vocab_occur = {}

    print("reading lines...")
    with open(LINES_PATH,"r",encoding="latin1") as f:
        lines_raw = f.read().split("\n")

    for line_raw in lines_raw:
        if line_raw != "":
            line_parts = line_raw.split(SPACER)
            text = line_parts[4].lower()
            lines[line_parts[0]] = text
            # While we're here, we minds as well
            # create a vocabulary for word embedding
            # some characters are separators in addition
            # to the space character " ".  We
            # handle that here:
            #tokens = word_tokenize(text)
            text = preTokenize(text)

            cleaned_text = re.sub(REGEX_SEARCH, ' ', text)
            tokens = cleaned_text.strip().split(" ")
            for t in tokens:
                # This will add duplicates, but
                # we remove them in the next step
                vocabulary.append(t)
                # Count occurences of each word
                if t not in vocab_occur:
                    vocab_occur[t] = 1
                else:
                    vocab_occur[t] += 1

    print("removing duplicates...")
    # Remove duplicates
    vocabulary = list(set(vocabulary))

    print("creating conversation objects...")
    with open(STRUCTURE_PATH,"r") as f:
        struct_raw = f.read().split("\n")
    for struct in struct_raw:
        if struct != "":
            subject1,subject2,movie_id,line_indices_raw = struct.split(SPACER)
            # line_indices_raw is a string of form
            # "['a','b',c']"
            # so we convert it to a list for manipulation
            line_indices = ast.literal_eval(line_indices_raw)
            conv_lines = [lines[l.strip()] for l in line_indices]
            convs.append(Conversation(subject1,subject2,conv_lines))

    if DEBUG:
        #for v in vocabulary:
        #    print(v)
        print(len(vocabulary))

        c = convs[0]
        print(c.lines)

        rare = []
        for k,v in vocab_occur.items():
            if v <= 1:
                rare.append(k)
        print(len(rare))

    if REMOVE_SINGLES:
        for k,v in vocab_occur.items():
            if v <= 1:
                vocabulary.remove(k)
    print(len(vocabulary))
    return vocabulary

if __name__=="__main__":
    parseDialog()
