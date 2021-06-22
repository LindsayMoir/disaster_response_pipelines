#!/usr/bin/env python
# coding: utf-8

# # train_classifier_command_line
# 
# This will run from the anaconda prompt via python train_classifier_command_line.py ..\data\drp.db ..\data\model.pkl.

# In[1]:


# import libraries
import nltk
nltk.download(['punkt', 'stopwords', 'wordnet'])
from nltk.corpus import stopwords
from nltk import download
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import numpy as np
import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

import sqlite3
import string
import sys
import time


# In[8]:


def load_data(database_filepath):
    
    """Read the input file, write the dataframe to sqlite, and create the independent and dependent arrays
    inputs:
    database_filepath: str: relative filepath on computer
    outputs:
    X: array: independent variable in this case is the messages column
    Y: array: binary values for each of the dependent features"""
    
    # Create connection with sqlite
    con = sqlite3.connect(database_filepath)
    
    # load data from database
    sql = "SELECT * FROM messages"
    df = pd.read_sql_query(sql, con) 
    
    # Need to speed this thing up. It is taking waaay too long. Just use a fraction of the rows.
    df = df.sample(frac=1)
    
    # Only want text column and dependent features 
    X = df['message'].values
    Y = df.iloc[:, 3:].values
    
    cols = df.columns[3:]

    return X, Y, cols, con


# In[9]:


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


# In[10]:


def build_gscv_models(X_train, Y_train, parameters):
    
    """Create model pipeline and return the GridSearchCV object
    inputs:
    X_train: array str: messages column
    Y_train: array binomial: 36 classification features
    parameters: dict: the items and range of parameters that we are searching
    outputs:
    cv: object: grid search object that contains what occurred and what is the best performing estimator"""
    
    # Create pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()), 
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # run grid search
    cv = GridSearchCV(pipeline, 
                      param_grid=parameters,
                      #n_jobs = -1,
                      verbose = 2)
    
    print('Training model for GridSearchCV ...')
    cv.fit(X_train, Y_train)
    
    # Select best model based on mean_test_score
    print('mean_test_score of best model is', np.max(cv.cv_results_['mean_test_score']))
    
    # print parameters for the best model
    print('Parameters for the best model are:\n', cv.best_params_)
    
    # return cv
    return cv


# In[11]:


def evaluate_model(y_true, y_pred, cols):
    
    """Calculate the accuracy, precision, recall, and F1 scores
    inputs:
    y_true: array binomial: of truth values
    y_pred: array binomial: of predicted values
    outputs:
    A dataframe of the accuracy, precision, recall, F1 scores"""
    
    # list of scoring functions
    functions = [accuracy_score, precision_score, recall_score, f1_score]
    
    # np.array to contain the results. There are 36 features and 4 scoring functions.
    results_array = np.zeros((36,4))
    
    # Get the shape of y_true's columns row to determine the width of the range (should be 36)
    for feature in range(y_true.shape[1]):
        
        # Inner loop to apply scoring functions
        for col, function in enumerate(functions):
            
            # rows are now columns. Think of where the features are.
            results_array[feature,col] = function((y_true)[:,feature], y_pred[:,feature])

    return pd.DataFrame(data = results_array, 
                        index = cols,
                        columns = ['accuracy', 'precision', 'recall','f1'])


# In[13]:


def main():

    """Function that drives the etl to completion
    inputs
    database_filepath: str: location of database on the computer in file system
    model_filepath: str: location of the model on the computer in file system
    outputs:
    Saves model to disk
    """
    
    # Get the start time
    start_time = time.time()

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))

         # Load data
        X, Y, category_names, con = load_data(database_filepath)

        # Split into train and test sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        # Instantiate parameters for grid search
        parameters = {'vect__min_df': [5],
                      'tfidf__use_idf':[False],
                      'clf__criterion': 'entropy',
                      'clf__estimator__n_estimators':[50, 100], 
                      'clf__estimator__min_samples_split':[5],
                      'clf__max_depth': 40,
                      'clf__max_features': 'auto'}

        print('Building model GridSearchCV and returning cv and best performing parameters based on X_train and Y_train ...')
        cv = build_gscv_models(X_train, Y_train, parameters)

        # The cv.best_esimator is ALREADY fit. No need to fit. Just predict
        print('Using best estimator to get predictions ...')
        # predict on test data
        model = cv.best_estimator_
        y_pred = model.predict(X_test)

        print('Scoring best estimator ...')
        score_df = evaluate_model(Y_test, y_pred, category_names)
        print(score_df)
        
        print('Writing scores to the db ...')
        # Write the df_score to the table scores
        score_df.to_sql('scores', con, if_exists = 'replace', index = False)
        
        # close the sqlite connection
        con.close()

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        pickle.dump(model, open('model.pkl', 'wb'))

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '              'as the first argument and the filepath of the pickle file to '              'save the model to as the second argument. \n\nExample: python '              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')
    
    # print how long this took
    print("\n--- %s minutes ---" % ((time.time() - start_time)/60))
    
    return score_df


# In[ ]:


if __name__ == '__main__':
    main()

