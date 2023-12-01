'''Functions for all the preprocessing of movies, users and ratings'''

import re
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')


scaler = MinMaxScaler()
def preprocess_movies(items_df):
    items_df['release_date'] = pd.to_datetime(items_df['release_date'], format='%d-%b-%Y')
    items_df['date'] = (items_df['release_date'].max() - items_df['release_date']).dt.days

    ## Fill missing values with mean
    items_df['days_ago'] = items_df['date'].fillna(items_df['date'].mean())

    def preprocess_title(title):
        remove_braces = re.sub(r'\([^)]*\)', '', title)
        if (len(remove_braces) == 0): remove_braces = title
        
        new_title = " ".join(re.findall(r'\b\w{3,}\b', remove_braces.lower())).strip()
        if (len(new_title) == 0): new_title = remove_braces

        return new_title    

    items_df['title'] = items_df['title'].apply(preprocess_title)
    
    tokenized_titles = [word_tokenize(title) for title in items_df['title']]
    vectors = Word2Vec(sentences=tokenized_titles, vector_size=50, window=5, min_count=1).wv

    def tokenize(title):
        # print(f'[{title}]')
        return vectors[word_tokenize(title)].mean(axis=0)

    items_df['title'] = items_df['title'].apply(tokenize)
    items_df = pd.concat([items_df, items_df['title'].apply(pd.Series)], axis=1)

    items_df['days_ago'] = scaler.fit_transform(items_df[['days_ago']])

    items_df = items_df.drop(columns=['video_release_date', 'imdb_url', 'release_date', 'date', 'title'])
    return items_df

def preprocess_users(users_df):
    users_df['zip_code'] = users_df['zip_code'].apply(lambda x: x[0])

    users_df['age'] = scaler.fit_transform(users_df[['age']])

    users_df = pd.get_dummies(users_df, columns=['occupation'], prefix='ocp', drop_first=True)
    users_df = pd.get_dummies(users_df, columns=['gender'], prefix='gender', drop_first=True)
    users_df = pd.get_dummies(users_df, columns=['zip_code'], prefix='zipcode', drop_first=True)

    return users_df

def preprocess_ratings(ratings_df):
    ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
    ratings_df['year'] = ratings_df['timestamp'].dt.year
    ratings_df['month'] = ratings_df['timestamp'].dt.month
    ratings_df['day'] = ratings_df['timestamp'].dt.day

    ratings_df.drop(columns=['timestamp'], inplace=True)

    ratings_df['rating'] = scaler.fit_transform(ratings_df[['rating']])
    ratings_df['year'] = scaler.fit_transform(ratings_df[['year']])
    ratings_df['month'] = scaler.fit_transform(ratings_df[['month']])
    ratings_df['day'] = scaler.fit_transform(ratings_df[['day']])

    return ratings_df

def merge(ratings_df, users_df, items_df):
    users = pd.merge(ratings_df, users_df, on='user_id')
    users_items = pd.merge(users, items_df, on='movie_id')
    return users_items