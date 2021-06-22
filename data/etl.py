#!/usr/bin/env python
# coding: utf-8

# In[70]:


# import libraries
import sys
import pandas as pd
import sqlite3


# In[71]:


def clean_categories(categories):
    """Loads categories csv and cleans it"""
    
    # Extract column names

    # Get the categories from the first row
    categories_str = categories.loc[0, 'categories']

    # split on ';'
    category_cols = categories_str.split(';')

    # Get the column names
    cols = [col.split('-')[0] for col in category_cols]
    
    # create a dataframe of the 36 individual category columns
    categories_encoded = categories['categories'].str.split(pat = ';', expand=True)
    
    # Rename the columns
    categories_encoded.columns = cols
    
    # Get rid of the alpha characters and only keep the one hot encoding (0 or 1)
    for col in cols:
        categories_encoded[col] = categories_encoded[col].str[-1:].astype(int)
    
    # Get the mode
    mode_ = categories_encoded['related'].mode()[0]
    
    # The request column has 3 unique labels [0,1,2]. Replace the 2 with the mode of the column (either 0 or 1)
    categories_encoded['related'] = categories_encoded['related'].replace(2, mode_)
        
    # Add the id column
    categories_encoded['id'] = categories['id']
    
    return categories_encoded


# In[72]:


def clean_merged(df):
    """Input is the merged df. Removes duplicates and drops a column that has a lot of NaNs"""
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    # calculate NaNs
    cols_null_mean = df.isnull().mean()
    
    # Find all columns with > 50% nulls
    cols_drop = cols_null_mean[cols_null_mean > .5].index.tolist()
        
    # drop the 'original column'. It is 62% NaN
    df.drop(columns = cols_drop, inplace = True)
    
    return df


# In[73]:


def load_db(df, db):
    """Load the dataframe into the sqlite database"""
    
    # Create connection (create db if not there)
    con = sqlite3.connect(db)
    
    # Write the df to the table messages
    df.to_sql('messages', con, if_exists = 'replace', index = False)
    
    # close the connection
    con.close()
    
    return


# In[74]:


def driver(file_1, file_2, db):
    
    """Function that drives the etl to completion
    inputs
    file_1: str: 'disaster_messages.csv'
    fil_2: str: 'disaster_categories.csv'
    db: str: 'drp.db'
    """
        
    print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'.format(file_1, file_2))
    # load messages dataset
    messages = pd.read_csv(file_1)

    # load categories dataset
    categories = pd.read_csv(file_2)

    print('Cleaning data...')
    # Clean categories
    categories = clean_categories(categories)

    # merge dataframes
    df = messages.merge(categories)

    # clean merged df
    df = clean_merged(df)

    print('Saving data...\n    DATABASE: {}'.format(db))
    # write to sqlite
    load_db(df, db)

    # status
    print('Cleaned data saved to database!')
    
    return


# In[75]:


def main():
    """Check and see if this is running in the command line or from Jupyter"""
    
    # Command line
    if len(sys.argv) == 4 sys.argv[0] == '-c':
        file_1, file_2, db = sys.argv[1:]
        driver(file_1, file_2, db)
    
    # Jupyter
    elif sys.argv[1] == '-f':
        driver('disaster_messages.csv', 'disaster_categories.csv', 'drp.db')
    
    # Wrong arguments
    else:
        print('Please provide the filepaths of the messages and categories '              'datasets as the first and second argument respectively, as '              'well as the filepath of the database to save the cleaned data '              'to as the third argument. \n\nExample: etl.py '              'disaster_messages.csv disaster_categories.csv '              'drp.db')
        
    return


# In[76]:


if __name__ == '__main__':
    main()


# In[ ]:




