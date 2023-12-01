'''script that performs evaluation of the final model'''

import pandas as pd
from src.data_utils import load_data, merge
from src.preprocess import preprocess_movies, preprocess_ratings, preprocess_users


def combine_with_all_movies(user_id, max_movie=1682, max_time=893286638, exclude_movies=set(), include_movies=None):
    '''Return a dataframe of a user combined with possible movies for prediction of rating'''
    data = []
    for m in range(1, max_movie+1):
        if ((include_movies is not None) and (m not in include_movies)): continue
        elif (m in exclude_movies): continue

        data.append([user_id, m, max_time])

    all_ratings = pd.DataFrame(data, columns=['user_id', 'movie_id', 'timestamp'])

    return all_ratings

# Produce test data
def get_for_test(base='u1'):
    '''Returns ready to predict on list of dataframes with the specif user in each combined with possible movies'''
    users, items, ratings_train = load_data(f'{base}.base')
    users, items, ratings = load_data(f'{base}.test')

    users, items = preprocess_users(users), preprocess_movies(items)

    test_users = set(ratings['user_id'])

    data_for_test = []
    for u in test_users:
        # Exclude movies of the user that were in train and include only test
        exclude_movies = set(ratings_train[ratings_train['user_id']==u]['movie_id'])
        include_movies = set(ratings[ratings['user_id']==u]['movie_id'])

        ratings_test = combine_with_all_movies(u, exclude_movies=exclude_movies, include_movies=include_movies)
        ratings_test = preprocess_ratings(ratings_test, has_rating=False)
        
        test_data = merge(ratings_test, users, items)

        data_for_test.append(test_data)

    return data_for_test

# BENCHMARK evluation function
def test_recall(base, k=20, liked_rating=4, model=None):
    '''
    Return average recall for user in the base test set
    
      Algorithm:
    - Pick a user
    - Iterate over test films and predict a rating
    - Pick top K predicted movies based on the rating
    - Calculate user recall: Check the present of each predicted film in the list of positively rated movies of this user
    - Iterate over all users and average the recall for all user in this test set
    '''
    recalls = []
    for test_user in get_for_test(base=base):
        X_test = test_user

        movie_ids = X_test['movie_id']
        user_id = X_test['user_id'][0]

        X_test = X_test.drop(columns=['user_id', 'movie_id'])

        # Predict rating for each film
        film_rating = [(m, r) for m, r in zip(movie_ids, model.predict(X_test))]
        film_rating = sorted(film_rating, key=lambda t: t[1], reverse=True)

        # Get top k rated films
        top_k = [m for m,_ in film_rating[:k]]

        # Get liked movies by the user
        _, _, ratings = load_data(f'{base}.test')
        liked_movies = list(ratings[(ratings['user_id']==user_id) & \
                                    (ratings['rating']>=liked_rating)]['movie_id'])   

        # Check how many predicted films are in the liked list
        hit = 0
        for m in top_k:
            if m in liked_movies: hit +=1
        
        # print(liked_movies)
        # print(film_rating[:k])
        recall = hit/len(liked_movies) if len(liked_movies) != 0 else 1
        # print(f"Recall for user {user_id}: {recall:.4f}")
        recalls.append(recall)
    return recalls