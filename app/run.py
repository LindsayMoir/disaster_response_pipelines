#!/usr/bin/env python
# coding: utf-8

# # Flask run.py

# In[65]:


# import libraries
import json
import nltk
# nltk.download(['punkt', 'stopwords', 'wordnet'])
from nltk.corpus import stopwords
from nltk import download
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request, jsonify
import pandas as pd
import pickle
import plotly
from plotly.graph_objs import Bar, Scatter
import sqlite3
import string


# In[66]:


# We had a problem with unpickling the model. Supposedly this will work. 
# That is taking out the function and importing it. It WORKED!
from tokenizer import tokenize


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

# data for first plot
genre_counts = df.groupby('genre').count()['message']
genre_names = list(genre_counts.index)

# Add up the number of messages per category in sorted order for the second plot.
df_2nd = df.iloc[:, 3:].sum().sort_values().reset_index()

# change column names to category and count
df_2nd.columns = ['category','count']

# Rename the first column to category for df_scores
df_scores.rename(columns = {df_scores.columns[0]: 'category'}, inplace=True)

# We want the same sort order for df_scores categories as for df_2nd.
# Append the count to df_scores via merge
df_scores = df_scores.merge(df_2nd, on = 'category')

# Sort df_scores by count
df_scores.sort_values('count', inplace = True)

# load model
model = pickle.load(open("../models/model.pkl", "rb"))


# In[69]:


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    """Does plotting for the web page. There are 3 plots. 2 bar plots and 1 line plot. 
    The line plot is based on the scoring of the model. DO NOT manipulate the data in this function.
    Do all data manipulations in the global area above. They often DO NOT carry over to here."""

    # create visuals
    
    graphs = [
        {
            'data': [
                Bar(
                    x = genre_names,
                    y = genre_counts
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
                    x = df_2nd['category'],
                    y = df_2nd['count'],
                    marker = dict(color='green')
                )
            ],

            'layout': {
                'title': "Numbers of Messages Per Message Category"
            }
        },
                {
            'data': [
                Scatter(
                    x = df_scores['category'],
                    y = df_scores['accuracy'],
                    name = 'Accuracy'
                ),
                Scatter(
                    x = df_scores['category'],
                    y = df_scores['precision'],
                    name = 'Precision'
                ),
                Scatter(
                    x = df_scores['category'],
                    y = df_scores['recall'],
                    name = 'Recall'
                ),
                Scatter(
                    x = df_scores['category'],
                    y = df_scores['f1'],
                    name = 'F1'
                )
            ],

            'layout': {
                'title': 'Evaluation Metrics For Each Category',
                'xaxis': {
                    'title':"",
                },
                'yaxis': {
                    'title': "Score",
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




