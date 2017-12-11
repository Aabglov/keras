# IMPORTS
import random
import os
import numpy as np # Used for One-hot encoding
import json # Used for json helper
import ast # USed to parse List from string ("['a','b',c']")

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(DIR_PATH,"data","dialog")
STRUCTURE_PATH = os.path.join(DATA_PATH,"movie_conversations.txt")
LINES_PATH = os.path.join(DATA_PATH,"movie_lines.txt")
SAVE_PATH = os.path.join(DATA_PATH,"parsed")
SPACER = " +++$+++ " # Arbitrary spacer token used by dataset

class Conversation:
    def __init__(self,subject1,subject2,lines):
        self.subject1 = subject1
        self.subject2 = subject2
        self.lines = lines

def tokenize(word):
    word = word.replace("-","")
    word = word.replace("\"","")
    word = word.replace("'","")
    word = word.replace("`","")
    word = word.replace(".","")
    word = word.replace(",","")
    word = word.replace(";","")
    word = word.replace(":","")
    word = word.replace("?","")
    word = word.replace("!","")
    word = word.replace("[","")
    word = word.replace("]","")
    word = word.replace("<u>","")
    word = word.replace("</u>","")
    word = word.replace("<a>","")
    word = word.replace("</a>","")
    word = word.replace("<b>","")
    word = word.replace("</b>","")
    word = word.replace("<i>","")
    word = word.replace("</i>","")
    word = word.replace("<","")
    word = word.replace(">","")
    #word = word.replace("&gt;","")
    #word = word.replace("&lt;","")
    word = word.replace("*","")
    word = word.strip()
    return word

#def parseDialog():
convs = []
lines = {}
vocabulary = []

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
        separated_text = text.replace("-"," ").replace("&"," ").replace("'","").replace("\t"," ")
        for word in separated_text.split(" "):
            # This will add duplicates, but
            # we remove them in the next step
            vocabulary.append(tokenize(word))

# Remove duplicates
vocabulary = list(set(vocabulary))

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


for v in vocabulary:
    print(v)
print(len(vocabulary))
