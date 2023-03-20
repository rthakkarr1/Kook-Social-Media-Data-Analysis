import pandas as pd
import json
import numpy as np
import re
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from nltk.util import ngrams
from moralizer import *

nltk.download('wordnet')
nltk.download('punkt')

df = pd.read_csv(r"C:\Users\12019\Desktop\Research Project Files\BTS Jungkook.csv")

to_drop = ['Date', 'Name', 'Shares', 'Location', 'Language']

df.drop(to_drop, inplace=True, axis=1)

def clean_text(text):
    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove emojis
    text = text.encode('ascii', 'ignore').decode('ascii')
    # Remove stop words
    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
 'youve', 'youll', 'youd', 'your', 'yours', 'yourself', 'yourselves', 'he',
 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it',
 "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',
 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and',
 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by',
 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now',
 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn',
 "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
 "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't",
 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn',
 "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn',
 "wouldn't",'rt']

    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word not in stop_words]
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    return ' '.join(lemmatized_words)

df['Message'] = df['Message'].apply(clean_text)



def mft_sentiment_analysis(text):
    words = word_tokenize(text)
    mft_df = pd.read_csv(r'C:\Users\12019\Desktop\Research Project Files\data.csv')
    categories = []

    for word in words:
        if word in mft_df['Word'].values:
            category = mft_df[mft_df['Word'] == word]['Category'].values[0]
            categories.append(category)

    return categories

df['Categories'] = df['Message'].apply(mft_sentiment_analysis)


df['Sentiment_Score'] = df['Message'].apply(mft_sentiment_analysis)

# Tokenize the 'Message' column
df['Tokenized_Message'] = df['Message'].apply(lambda x: word_tokenize(x))

# Combine all tokenized messages into one list
all_words = [word for tokenized_message in df['Tokenized_Message'] for word in tokenized_message]

# Create frequency distribution of all words
fdist = FreqDist(all_words)

# Print the 10 most common words
print(fdist.most_common(20))


# Convert tokenized messages back to strings
df['Text'] = df['Tokenized_Message'].apply(lambda x: ' '.join(x))

# Vectorize text using CountVectorizer
vectorizer = CountVectorizer(max_features=10000, max_df=0.95, min_df=2, stop_words='english')
X = vectorizer.fit_transform(df['Text'])

# Fit LDA model
lda = LatentDirichletAllocation(n_components=10, random_state=42)
lda.fit(X)

# Print top 10 words in each topic
for i, topic in enumerate(lda.components_):
    print(f"Top 10 words for topic {i}")
    print([vectorizer.get_feature_names_out()[idx] for idx in topic.argsort()[-10:]])
    print('\n')

# Extract 5-grams from tokenized messages
df['FiveGrams'] = df['Tokenized_Message'].apply(lambda x: list(ngrams(x, 5)))

# Combine all 5-grams into one list
all_fivegrams = [fivegram for message_fivegrams in df['FiveGrams'] for fivegram in message_fivegrams]

# Create frequency distribution of all 5-grams
fdist_fivegrams = FreqDist(all_fivegrams)

# Extract unique 5-grams
unique_fivegrams = set(all_fivegrams)

# Print the number of unique 5-grams
print("Number of unique 5-grams:", len(unique_fivegrams))

# Print the 10 most common 5-grams
print(fdist_fivegrams.most_common(20))

# Save results to a CSV file
result_df = df.copy()
topic_word_counts = lda.transform(X).sum(axis=0)
fivegram_count = sum([len(x) for x in result_df['FiveGrams'].tolist()])

freq_data = {'Top 10 Topic Frequency': [topic_word_counts],
             'Top 10 FiveGram Frequency': [fivegram_count]}

freq_df = pd.DataFrame(freq_data)
freq_df.to_csv(r'C:\Users\12019\Desktop\Research Project Files\frequency_data.csv', index=False)

result_df['Topic'] = np.argmax(lda.transform(X), axis=1)

result_df[['Message', 'Categories', 'Topic']].to_csv(r'C:\Users\12019\Desktop\Research Project Files\sentiment_analysis_outputkook4.csv', index=False)

