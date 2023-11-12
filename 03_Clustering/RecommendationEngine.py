import numpy as np
import json

"""
Movie Recommendation Engine

Installation:
Assuming that you have pip installed, type this in a terminal: sudo pip install numpy

Overview:
Basic movie recommendation engine showing 5 movies to recommend and 5 to avoid based on data given in .json file for 
selected user

Authors:
By Maciej Zagórski (s23575) and Łukasz Dawidowski (s22621), group 72c (10:15-11:45).

Usage:
Define target user u want to recommend movies (the user must be defined in .json file in order for code to work properly
and must have defined some movie ratings)
Default is 5 movies to recommend and five to avoid you can change it by changing "num_recommendations" parameter in 
"recommend_movies" function 
"""


def load_user_ratings(file_path):
    """
       Load user ratings from a JSON file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def create_ratings_matrix(user_ratings, movies):
    """
        Create a ratings matrix from user ratings.
    """
    num_users = len(user_ratings)
    num_movies = len(movies)

    ratings_matrix = np.zeros((num_users, num_movies))

    for i, (user, ratings) in enumerate(user_ratings.items()):
        for j, movie in enumerate(movies):
            if movie in ratings:
                ratings_matrix[i, j] = ratings[movie]

    return ratings_matrix


def calculate_correlation(ratings_matrix, target_user_ratings):
    """
        Calculate correlation scores between the target user and other users.
    """
    correlation_scores = np.corrcoef(ratings_matrix, target_user_ratings)
    return correlation_scores[-1, :-1]


def recommend_movies(user_ratings, target_user, num_recommendations=5):
    """
        Recommend movies for a target user based on collaborative filtering.

        Args:
            user_ratings (dict): A dictionary representing user ratings.
            target_user (str): The target user for whom recommendations are made.
            num_recommendations (int): Number of movies to recommend.

        Returns:
            tuple: A tuple containing lists of recommended movies and movies to avoid.
    """
    users = list(user_ratings.keys())
    movies = list(set(movie for ratings in user_ratings.values() for movie in ratings.keys()))

    ratings_matrix = create_ratings_matrix(user_ratings, movies)

    target_user_ratings = ratings_matrix[users.index(target_user)]
    correlation_with_target = calculate_correlation(ratings_matrix, target_user_ratings)

    similar_users = np.argsort(correlation_with_target)[::-1]

    target_user_unwatched_movies = [movie for movie in movies if movie not in user_ratings[target_user]]

    predicted_ratings = np.dot(ratings_matrix[similar_users[:5]].T,
                               correlation_with_target[similar_users[:5]]) / np.sum(
        np.abs(correlation_with_target[similar_users[:5]]))

    recommended_movies = [movie for movie in target_user_unwatched_movies if
                          predicted_ratings[movies.index(movie)] > np.mean(predicted_ratings)]
    recommended_movies = sorted(recommended_movies, key=lambda x: predicted_ratings[movies.index(x)], reverse=True)[
                         :num_recommendations]

    avoid_movies = [movie for movie in target_user_unwatched_movies if
                    predicted_ratings[movies.index(movie)] < np.mean(predicted_ratings)]
    avoid_movies = sorted(avoid_movies, key=lambda x: predicted_ratings[movies.index(x)])[:num_recommendations]

    return recommended_movies, avoid_movies


if __name__ == "__main__":
    file_path = 'film_data.json'
    target_user = "Łukasz Dawidowski"
    user_ratings = load_user_ratings(file_path)
    recommendations, avoidances = recommend_movies(user_ratings, target_user)

    print(f"\nRecommended Movies for {target_user}:")
    for movie in recommendations:
        print(movie)

    print(f"\nMovies to Avoid for {target_user}:")
    for movie in avoidances:
        print(movie)
