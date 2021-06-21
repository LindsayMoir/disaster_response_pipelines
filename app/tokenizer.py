#!/usr/bin/env python
# coding: utf-8

# # Tokenize

# In[ ]:


# import libraries
import json
import nltk
nltk.download(['punkt', 'stopwords', 'wordnet'])
from nltk.corpus import stopwords
from nltk import download
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string


# In[ ]:


def tokenize(text):

    """Tokenize text
    inputs:
    text: array str: messages column
    outputs:
    tokens: list: list of cleaned up tokens suitable for nlp
    """
        
    # Lower case
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # tokenize text
    tokens = word_tokenize(text)
    
    # Remove stop words
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    
    # lemmatize tokens
    tokens = [WordNetLemmatizer().lemmatize(token) for token in tokens] # pos nouns are default
    tokens = [WordNetLemmatizer().lemmatize(token, pos = 'v') for token in tokens]

    # strip white space from tokens
    tokens = [token.strip() for token in tokens]

    return tokens


# In[ ]:


if __name__ == '__main__':
    tokenize(text)


# In[ ]:




