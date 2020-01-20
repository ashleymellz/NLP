# sentiment_analysis.py
# This script can take in a csv/txt file with one column of text, and assign
# a sentiment value of negative, positive, or neutral

# The script will spit out a new csv with the sentiment in a new column

__author__ = "Ashley Melanson"
__version__ = "1.0.1"

import csv
import re

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


#################################################################################
# GLOBAL CONSTANTS
#################################################################################

IN_FILENAME = 'test_nlp.txt'
OUT_FILENAME = 'test_output.csv'

# Normalization
lemmatizer = WordNetLemmatizer()

#################################################################################
# FUNCTION DEFINITIONS
#################################################################################

class TextCleanup(object):
    ''' some functions for the purpose of cleaning up text '''
    def __init__(self):
        pass

    def remove_whitespace(self):
        cleaned = " ".join(re.split("\s+", self, flags=re.UNICODE))
        return cleaned

    def remove_punctuation(self):
        return ''

    def remove_stopwords(self):
        return ''


class WhitespaceTokenizer(object):
    ''' Split the text into words '''
    def __init__(self):
        self.tokenizer = RegexpTokenizer('\s+', gaps=True)

    def split(self, text):
        tokens = self.tokenizer.tokenize(text)
        return tokens


class LemmatizerWithTagger(object):
    def __init__(self):
        pass

    def get_wordnet_pos(self, tag):
        ''' return the proper wordnet pos tag '''
        if tag.startswith('J'):
            # POS tags : JJ, JJR, JJS
            return wordnet.ADJ
        elif tag.startswith('N'):
            # POS tags : NN, NNS, NNP, NNPS
            return wordnet.NOUN
        elif tag.startswith('R'):
            # POS tags : RB, RBR, RBS
            return wordnet.ADV
        elif tag.startswith('V'):
            # POS tags : VB, VBD, VBG, VBN, VBP, VBZ
            return wordnet.VERB
        else:
            return wordnet.NOUN

    def pos_tag(self, tokens):
        # find the pos tag for each token produced by nltk
        pos_tokens = [nltk.pos_tag(tokens) for token in tokens]

        # lemmatization using WordNetLemmatizer
        pos_tokens = [ [(word, lemmatizer.lemmatize(word, self.get_wordnet_pos(pos_tag)), [pos_tag]) for (word, pos_tag) in pos] for pos in pos_tags]
        return pos_tokens

## MAIN CALLS #########################################################

tokenizer = custom_tokenizer()
lemmatizer = LemmatizerWithTagger()

text = "I hated apples yesterday and today I am loving oranges"

token_text = tokenizer.split(text)

tagged_text = lemmatizer.pos_tag(token_text)

print(tagged_text)


#################################################################################
# MAIN CALLS
#################################################################################

with open(IN_FILENAME, mode='r', encoding='utf-8') as infile:
    reader = csv.DictReader(infile, delimiter='\t')
    sample_data = list(reader)

for dct in sample_data:
    manipulate(dct, 'responses', 'words')

with open(OUT_FILENAME, mode='w') as outfile:
    fieldnames = ['responses','words']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in sample_data:
        writer.writerow(row)
