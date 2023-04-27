import pandas as pd

# Function to format the csv file
def formatCSV(file):
    # Read in file
    df = pd.read_csv(file, header=None)

    # Add in column names
    df.rename(columns={0: 'ids', 1: 'user', 2: 'sentiment score', 3: 'text'}, inplace=True)

    # Drop unecessary columns
    df = df.drop(columns=['ids','user'])

    # Remove irrelevant rows
    df.drop(df[df["sentiment score"] == "Irrelevant"].index, inplace=True)

    # Convert sentiment from string to numbers
    df['sentiment'] = df['sentiment score'].replace(["Negative", "Neutral", "Positive"], [0,1,2])

    # Switch the order of the columns so they are in the correct order
    df = df[['text', 'sentiment']]

    # Add a max sentiment column
    df['sentimentMax'] = 2

    return df

# Grab the formated dataframes
training = formatCSV("twitter_training.csv")


validation = formatCSV("twitter_validation.csv")


# Saving the dataframes to csv's
training.to_csv('preprocessed_training.csv', index=False)

validation.to_csv('preprocessed_validation.csv', index=False)