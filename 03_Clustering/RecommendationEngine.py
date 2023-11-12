import numpy as np
import json

with open('film_data.json', 'r', encoding='utf-8') as file:
    user_ratings = json.load(file)

def recommend_movies(user_ratings, target_user, num_recommendations=5):
    users = list(user_ratings.keys())
    movies = list(set(movie for ratings in user_ratings.values() for movie in ratings.keys()))
    num_users = len(users)
    num_movies = len(movies)

    ratings_matrix = np.zeros((num_users, num_movies))

    for i, user in enumerate(users):
        for j, movie in enumerate(movies):
            if movie in user_ratings[user]:
                ratings_matrix[i, j] = user_ratings[user][movie]

    target_user_ratings = ratings_matrix[users.index(target_user)]
    correlation_scores = np.corrcoef(ratings_matrix, target_user_ratings)

    correlation_with_target = correlation_scores[-1, :-1]

    similar_users = np.argsort(correlation_with_target)[::-1]

    target_user_unwatched_movies = [movie for movie in movies if movie not in user_ratings[target_user]]

    predicted_ratings = np.dot(ratings_matrix[similar_users[:5]].T, correlation_with_target[similar_users[:5]]) / np.sum(np.abs(correlation_with_target[similar_users[:5]]))

    recommended_movies = [movie for movie in target_user_unwatched_movies if predicted_ratings[movies.index(movie)] > np.mean(predicted_ratings)]
    recommended_movies = sorted(recommended_movies, key=lambda x: predicted_ratings[movies.index(x)], reverse=True)[:num_recommendations]

    avoid_movies = [movie for movie in target_user_unwatched_movies if predicted_ratings[movies.index(movie)] < np.mean(predicted_ratings)]
    avoid_movies = sorted(avoid_movies, key=lambda x: predicted_ratings[movies.index(x)])[:num_recommendations]

    return recommended_movies, avoid_movies

target_user = "Paweł Czapiewski"
recommendations, avoidances = recommend_movies(user_ratings, target_user)

print(f"\nRecommended Movies for {target_user}:")
for movie in recommendations:
    print(movie)

print(f"\nMovies to Avoid for {target_user}:")
for movie in avoidances:
    print(movie)
