import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk # NLTK:is used for understanding of human natural language.
import io
import unicodedata
import numpy as np
import re
import string
from numpy import linalg
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import webtext
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('vader_lexicon')



with open('/Users/apple/Desktop/Data_science_projects/Sentiment_Analysis_NLP/all_kindle_review.csv', encoding ='ISO-8859-2') as f:
	text = f.read()


# tokenize the data

sent_tokenizer = PunktSentenceTokenizer(text)
sents = sent_tokenizer.tokenize(text)

print(word_tokenize(text))
print(sent_tokenize(text))

# stemize the data

porter_stemmer = PorterStemmer()

nltk_tokens = nltk.word_tokenize(text)

for w in nltk_tokens:
	print ("Actual: % s Stem: % s" % (w, porter_stemmer.stem(w)))
	
# lemmatize the data
wordnet_lemmatizer = WordNetLemmatizer()
nltk_tokens = nltk.word_tokenize(text)

for w in nltk_tokens:
	print ("Actual: % s Lemma: % s" % (w, wordnet_lemmatizer.lemmatize(w)))

# POS( part of speech) tagging of the tokens and select only significant features/tokens like adjectives, 
# adverbs, and verbs, etc.	
text = nltk.word_tokenize(text)
print(nltk.pos_tag(text))

sid = SentimentIntensityAnalyzer() 
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# used the polarity_scores() method to obtain the polarity indices for the given sentence
with open('/Users/apple/Desktop/Data_science_projects/Sentiment_Analysis_NLP/all_kindle_review.csv', encoding ='ISO-8859-2') as f:
	for text in f.read().split('\n'):
		print(text)
		scores = sid.polarity_scores(text)
		for key in sorted(scores):
			print('{0}: {1}, '.format(key, scores[key]), end ='')
			
	print()

# The Compound score is a metric that calculates the sum of all the lexicon ratings which have been 
# normalized between -1( extreme negative) and +1 ( extreme positive).