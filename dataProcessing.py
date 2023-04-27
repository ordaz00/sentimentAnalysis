import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#function to format the csv file
def formatCSV(file):
    #read in file
    df = pd.read_csv(file, header=None)

    #add in column names
    df.rename(columns={0: 'ids', 1: 'user', 2: 'target', 3: 'text'}, inplace=True)

    #drop unecessary columns
    df = df.drop(columns=['ids','user'])

    #remove irrelevant rows
    df.drop(df[df["target"] == "Irrelevant"].index, inplace=True)

    #convert sentiment from string to numbers
    df['target'] = df['target'].replace(["Negative", "Neutral", "Positive"], [0,1,2])

    return df


