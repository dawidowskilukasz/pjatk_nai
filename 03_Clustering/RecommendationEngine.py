"""
Movie Recommendation (and Anti-Recommendation) Engine

Installation:
Assuming that you have pip installed, type this in a terminal: sudo pip install numpy (with regard to arrays and
math-like functions used in the code)

Overview:
This is a basic movie recommendation (and anti-recommendation) engine, providing 5 movies recommended to watch and 5
movies to avoid for a selected (target) user, based on the data given in JSON file (film_data.json). The engin uses
different distance metrics to find correlations between the film tastes of different people and their films ratings.

The distance metrics functions have been slightly modified, by adding „bonus” to them, in order to value more ratings of
people who rated more the same movies as the target user. If a particular person does not share any rated movie as the
target user, his/her rating will be considered as netrual (movies watched by this person films watched by that person
should not be regarded as either recommended or not recommended – it is not known).

Authors:
By Maciej Zagórski (s23575) and Łukasz Dawidowski (s22621), group 72c (10:15-11:45).

Sources:
https://numpy.org/ (NumPy documentation)

Usage:
- Define the target user to recommend movies. The user must be defined in JSON file in order for code to work properly
  and must have defined some movie ratings.
- The distance metric may be chagned using „distance_metrics” parameter in the „recommend_movies” function; as for now,
  it is possible to chose between the Euclidean distance ('Euclidean'; the default value), the Manhattan distance
  ('Manhattan') and the Pearson's distance ('Pearson').
- Number of movies to recommend and to avoid can be changed by setting-up the „num_recommendations” parameter in
  the „recommend_movies” function (the defalut value is 5).
"""

import numpy as np
import json
import sys


def load_users_ratings(file_path):
    """
        Load users ratings from the JSON file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def create_ratings_matrix(users_ratings, movies):
    """
        Create the ratings matrix from the users ratings.
    """
    num_users = len(users_ratings)
    num_movies = len(movies)

    ratings_matrix = np.zeros((num_users, num_movies))

    for i, (user, ratings) in enumerate(users_ratings.items()):
        for j, movie in enumerate(movies):
            if movie in ratings:
                ratings_matrix[i, j] = ratings[movie]

    return ratings_matrix


# def calculate_correlation(ratings_matrix, target_user_ratings):
#     """
#     Calculate correlation scores between the target user and other users.
#     """
#     correlation_scores = np.corrcoef(ratings_matrix, target_user_ratings)
#     return correlation_scores[-1, :-1]


def euclidean_score_with_bonus(user_movies, target_user_movies):
    """
        Calculate the correlation score between the target user and the other user using Euclidean distance.

        The num_ratings multiplication at the end is used in order to value more ratings of people who rated more the
        same movies as the target user.
    """
    common_movies = {key: 1 for key in user_movies.keys() if key in target_user_movies}

    num_ratings = len(common_movies)

    if num_ratings == 0:
        return 0

    squared_diff = np.square(np.array([user_movies[key] - target_user_movies[key] for key in common_movies]))

    return 1 / (1 + np.sqrt(np.sum(squared_diff))) * num_ratings


def manhattan_score_with_bonus(user_movies, target_user_movies):
    """
        Calculate the correlation score between the target user and the other user using Manhattan distance.

        The num_ratings multiplication at the end is used in order to value more ratings of people who rated more the
        same movies as the target user.
    """
    common_movies = {key: 1 for key in user_movies.keys() if key in target_user_movies}

    num_ratings = len(common_movies)

    if num_ratings == 0:
        return 0

    abs_diff = np.abs(np.array([user_movies[key] - target_user_movies[key] for key in common_movies]))

    return np.sum(abs_diff) * num_ratings


def pearson_score_with_bonus(user_movies, target_user_movies):
    """
        Calculate the correlation score between the target user and the other user using Pearson distance.

        The num_ratings multiplication at the end is used in order to value more ratings of people who rated more the
        same movies as the target user
    """
    common_movies = {key: 1 for key in user_movies.keys() if key in target_user_movies}

    num_ratings = len(common_movies)

    if num_ratings == 0:
        return 0

    user_ratings = np.array([user_movies[key] for key in common_movies])
    target_user_ratings = np.array([target_user_movies[key] for key in common_movies])

    Sxy = np.sum(user_ratings * target_user_ratings) - np.sum(user_ratings) * np.sum(target_user_ratings) / num_ratings
    Sxx = np.sum(np.square(user_ratings)) - np.square(np.sum(user_ratings)) / num_ratings
    Syy = np.sum(np.square(target_user_ratings)) - np.square(np.sum(target_user_ratings)) / num_ratings

    if Sxx * Syy == 0:
        return 0

    return Sxy / np.sqrt(Sxx * Syy) * num_ratings


def recommend_movies(user_ratings, target_user, distance_metrics='euclidean', num_recommendations=5):
    """
        Recommend movies for a target user based on collaborative filtering.

        Args:
            users_ratings (dict): A dictionary representing the users ratings.
            target_user (str): The target user for whom recommendations (and anti-recommendations) are made.
            distance_metrics (str): The distance metric to find correlations between users ratings.
            num_recommendations (int): Number of movies to recommend and to anti-recommend.

        Returns:
            tuple: A tuple containing lists of movies recommended to watch and movies to avoid.
    """
    users = list(user_ratings.keys())

    if target_user not in users:
        sys.exit("Error! User not found!")

    movies = list(set(movie for ratings in user_ratings.values() for movie in ratings.keys()))

    ratings_matrix = create_ratings_matrix(user_ratings, movies)

    # target_user_ratings = ratings_matrix[users.index(target_user)]

    if distance_metrics == 'Euclidean':
        correlation_with_target = np.array([
            euclidean_score_with_bonus(user_ratings[key], user_ratings[target_user]) for key in user_ratings
        ])
    elif distance_metrics == 'Pearson':
        correlation_with_target = np.array([
            pearson_score_with_bonus(user_ratings[key], user_ratings[target_user]) for key in user_ratings
        ])
    elif distance_metrics == 'Manhattan':
        correlation_with_target = np.array([
            manhattan_score_with_bonus(user_ratings[key], user_ratings[target_user]) for key in user_ratings
        ])
    # correlation_with_target = calculate_correlation(ratings_matrix, target_user_ratings)

    similar_users = np.argsort(correlation_with_target)[::-1]

    target_user_unwatched_movies = [movie for movie in movies if movie not in user_ratings[target_user]]

    predicted_ratings = np.dot(ratings_matrix[similar_users[1:6]].T,
                               correlation_with_target[similar_users[1:6]]) / np.sum(
        np.abs(correlation_with_target[similar_users[1:6]]))

    recommended_movies = [movie for movie in target_user_unwatched_movies if
                          predicted_ratings[movies.index(movie)] > np.mean(predicted_ratings)]
    recommended_movies = sorted(recommended_movies, key=lambda x: predicted_ratings[movies.index(x)], reverse=True)[
                         :num_recommendations]

    avoid_movies = [movie for movie in target_user_unwatched_movies if
                    np.mean(predicted_ratings) > predicted_ratings[movies.index(movie)] > 0]
    avoid_movies = sorted(avoid_movies, key=lambda x: predicted_ratings[movies.index(x)])[:num_recommendations]

    return recommended_movies, avoid_movies


if __name__ == "__main__":
    file_path = 'film_data.json'
    target_user = ['Maciej Zagórski']
    distance_metrics = ['Euclidean', 'Manhattan', 'Pearson']

    users_ratings = load_users_ratings(file_path)

    for user in target_user:
        for metrics in distance_metrics:
            recommendations, avoidances = recommend_movies(users_ratings, user, metrics)

            print(f"\nRecommended Movies for {user} using {metrics} distance:")
            for movie in recommendations:
                print(movie)

            print(f"\nMovies to Avoid for {user} using {metrics} distance:")
            for movie in avoidances:
                print(movie)
