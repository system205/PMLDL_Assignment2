'''Extract and load data'''

import zipfile
import os
import pandas as pd


def unzip(file = 'data/raw/ml-100k.zip', \
          target_dir = 'data/interim/'):
    

    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(target_dir)

    print(f"Successfully extracted to {target_dir}")


def load_data(data_file, ml_100k_folder = 'data/interim/ml-100k/'):
    user_file = 'u.user'
    item_file = 'u.item'
    genre_file = 'u.genre'

    # column names
    user_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    data_cols = ['user_id', 'movie_id', 'rating', 'timestamp']

    # Load data into Pandas DataFrames
    users = pd.read_csv(os.path.join(ml_100k_folder, user_file), sep='|', names=user_cols)
    data = pd.read_csv(os.path.join(ml_100k_folder, data_file), sep='\t', names=data_cols)

    genre = pd.read_csv(os.path.join(ml_100k_folder, genre_file), sep='|', header=None, names=['genre_id', 'genre'])

    item_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url', *[g[0] for g in genre.values]]
    items = pd.read_csv(os.path.join(ml_100k_folder, item_file), sep='|', names=item_cols, encoding='latin-1')

    return users, items, data
