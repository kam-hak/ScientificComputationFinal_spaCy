import pandas as pd
import spacy
from spacy.lang.en import English
from spacy import displacy
from spacy.tokens import DocBin
from sklearn.model_selection import train_test_split

# Reading in csv as panda df
df = pd.read_csv("XR_tweets.csv")

# first rows
print(df.head())

# double check data type
print(type(df))


# spacy's 'small' core english language model

#!python -m spacy download en_core_web_trf

# Assumes you have downloaded en_core_web_sm via terminal: 'python -m spacy download en_core_web_sm'
nlp = spacy.load("en_core_web_sm")

# filtering for only english tweets and is_retweet == False
en_df = df[(df["language"] == "en") & (df["is_retweet"] == False)]

# language breakdown
print(len(df[df["language"] == "en"]) / len(df) * 100)
print(len(en_df) / len(df[df["language"] == "en"]) * 100)
print(len(en_df) / len(df) * 100)

## export sample of tweets to manually look over to identify hashtags
sample_tweets = en_df.sample(10000)
sample_tweets.to_csv("sample_tweets.csv", index=False)


## Defining binary confidence score, addingto en_df
def confidence_score(hashtags):
    certain_hashtags = [
        "rebel",
        "extinct",
        "scientistrebel",
        "disobedi",
        "xr",
        "scientist",
    ]
    hashtags_str = str(hashtags).lower()
    for certain_tag in certain_hashtags:
        if certain_tag in hashtags_str:
            return 1.0
    return 0.0


en_df["confidence_score"] = en_df["hashtags"].apply(confidence_score)

# # pretraining

# # Read in the csv
# df = pd.read_csv("csv_path")

# # Prepare the raw text dataset
# pretrain_docbin = DocBin()
# for _, row in df.iterrows():
#     text = row["text"]

#     doc = nlp.make_doc(text)
#     pretrain_docbin.add(doc)

# # Create the .spacy file
# pretrain_docbin.to_disk("./raw_text.spacy")
