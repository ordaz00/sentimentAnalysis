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

# Function to preprocess the text
def preprocess_text(text):
   # Handle non-string values
   if not isinstance(text, str):
       return ""
  
   # Remove URLs
   text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)


   # Remove special characters, numbers, and punctuation
   text = re.sub(r'\W', ' ', text)


   # Remove extra spaces
   text = re.sub(r'\s+', ' ', text)


   # Convert to lowercase
   text = text.lower()


   # Tokenize the text
   tokens = nltk.word_tokenize(text)


   # Remove stop words
   stop_words = set(stopwords.words('english'))


   tokens = [word for word in tokens if word not in stop_words]


   # Lemmatize the tokens
   lemmatizer = WordNetLemmatizer()


   tokens = [lemmatizer.lemmatize(word) for word in tokens]


   # Join the tokens back into a single string
   text = ' '.join(tokens)


   return text
