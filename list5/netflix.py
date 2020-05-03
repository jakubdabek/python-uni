from list5 import toy_story
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
import numpy as np


def main():
    movies = toy_story.load_data("movies.csv")
    movie_ids = movies["movieId"]
    movie_ids = movie_ids[movie_ids < 10_000]
    ratings = toy_story.load_data("ratings.csv")
    data = toy_story.relevant_data(ratings, movie_ids, None, 10_000)

    my_ratings = {
        2571: 5.0,
        32: 4.0,
        260: 5.0,
        1097: 4.0,
    }
    my_ratings = pd.DataFrame(dict(rating=my_ratings), index=movie_ids)
    my_ratings = my_ratings.fillna(0.0)

    profile = cosine_similarity(data, my_ratings.T)
    profile = pd.DataFrame(index=data.index, columns=["cos(theta)"], data=profile)

    recommendations = cosine_similarity(data.T, profile.T)
    recommendations = pd.DataFrame(index=movie_ids, columns=["recommendation"], data=recommendations)

    recommendations = pd.merge(recommendations, movies[["movieId", "title"]], on="movieId")
    recommendations.set_index("movieId", inplace=True)
    recommendations.sort_values("recommendation", ascending=False, inplace=True)

    return movies, data, my_ratings, profile, recommendations


def main_print():
    movies, data, my_ratings, profile, recommendations = main()
    print(recommendations.head(10))


if __name__ == '__main__':
    main_print()
