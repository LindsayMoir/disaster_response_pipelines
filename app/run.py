#!/usr/bin/env python
# coding: utf-8

# # Flask run.py

# In[65]:


# import libraries
import json
import nltk
nltk.download(['punkt', 'stopwords', 'wordnet'])
from nltk.corpus import stopwords
from nltk import download
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request, jsonify
import pandas as pd
import pickle
import plotly
from plotly.graph_objs import Bar, Line
import sqlite3
import string


# In[66]:


# We had a problem with unpickling the model. Supposedly this will work. 
# That is taking out the function and importing.
import tokenize


# In[67]:


# instantiate flask
app = Flask(__name__)


# In[68]:


# load data
con = sqlite3.connect('../data/drp.db')

# Get messages
sql = "SELECT * FROM messages"
df = pd.read_sql_query(sql, con)

# Get scores
sql = "SELECT * FROM scores"
df_scores = pd.read_sql_query(sql, con)

# close the sqlite connection
con.close()

# Rename the first column to category
df_scores.rename(columns = {df_scores.columns[0]: 'category'}, inplace=True)

# Melt the df_scores to make it easy to do a line plot
df_melt = df_scores.melt(id_vars = 'category', 
                         value_vars = ['accuracy', 'precision', 'recall', 'f1'],
                         var_name = 'metric',
                         value_name = 'score')

# load model
model = pickle.load(open("../models/model.pkl", "rb"))


# In[69]:


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    """Does plotting for the web page. There are 3 plots. 2 bar plots and 1 line plot. 
    The line plot is based on the scoring of the model"""
    
    # Add up the number of messages per category in sorted order for the second plot.
    df_2nd = df.iloc[:, 3:].sum().sort_values().reset_index()
    
    # change column names to Category and Count
    df_2nd.columns = ['category','count']

    # create visuals

    
    graphs = [
        {
            'data': [
                Bar(
                    x = list(genre_counts.index),
                    y = df.groupby('genre').count()['message']
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x = df_2nd['count'],
                    y = df_2nd['category'],
                    orientation = 'h',
                    marker = dict(color='green')
                )
            ],

            'layout': {
                'title': "Numbers of Messages Per Message Category",
                'xaxis': {
                    'title':"Message Numbers"
                }
            }
        },
        {
            'data': [
                Line(
                    x = df_melt['category'],
                    y = df_melt['score'],
                    color = df_melt['metric']
                )
            ],

            'layout': {
                'title': 'Accuracy, Precision, Recall, and F1 Scores For The Classification Model',
                'xaxis': {
                    'title':"Categories",
                },
                'yaxis': {
                    'title': "Score"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# In[70]:


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


# In[71]:


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)
    


# In[ ]:


if __name__ == '__main__':
    main()


# In[ ]:




