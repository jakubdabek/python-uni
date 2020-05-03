from pathlib import Path
from typing import Union, Optional, Tuple
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from list4.overload import flatten

import pandas as pd
import numpy as np


def load_data(file_name: str, *, dir_path: Optional[Union[Path, str]] = None) -> pd.DataFrame:
    if dir_path is None:
        dir_path = Path(__file__).parent / 'ml-latest-small'
    elif not isinstance(dir_path, Path):
        dir_path = Path(dir_path)

    return pd.read_csv(dir_path / file_name)


def load_movies_find_id(title: str) -> Tuple[pd.Series, int]:
    movies = load_data("movies.csv")
    return movies["movieId"], movies.loc[movies["title"] == title, "movieId"][0]


def relevant_data(ratings: pd.DataFrame, movie_ids: pd.Series, movie_id: Optional[int], max_movie_id: int) -> pd.DataFrame:
    ratings = ratings.drop('timestamp', axis=1)
    ratings = ratings[ratings['movieId'] <= max_movie_id]
    ratings = pd.pivot_table(ratings, index='userId', columns='movieId', values='rating')
    ratings = ratings.reindex(columns=movie_ids[movie_ids <= max_movie_id], copy=False)
    if movie_id is not None:
        ratings = ratings[ratings[movie_id].notna()]
    ratings.fillna(0.0, inplace=True)

    return ratings


def linear_regression_dataset(ratings: pd.DataFrame, movie_id: int, *, inplace=False) -> Tuple[pd.DataFrame, pd.Series]:
    movie_ratings = ratings[movie_id]
    return ratings.drop(movie_id, axis=1, inplace=inplace), movie_ratings


def main():
    movie_ids, toy_story_id = load_movies_find_id("Toy Story (1995)")
    ratings = load_data("ratings.csv")

    max_movie_ids = pd.Series(flatten([10**i, 10**i * 2, 10**i * 5] for i in range(1, 6)))
    # max_movie_ids = [200]
    # max_movie_ids = [10, 100, 200, 500, 1000, 2500, 5000, 7500, 10000]
    bound = len(max_movie_ids[max_movie_ids < movie_ids.max()])
    max_movie_ids = max_movie_ids[:bound+1]
    scores = []
    for max_movie_id in max_movie_ids:
        data = relevant_data(ratings, movie_ids, toy_story_id, max_movie_id)
        (_, toy_story) = linear_regression_dataset(data, toy_story_id, inplace=True)
        lr: LinearRegression = LinearRegression().fit(data, toy_story)
        scores.append(lr.score(data, toy_story))

        lr: LinearRegression = LinearRegression().fit(data[:-15], toy_story[:-15])
        predictions: np.ndarray = lr.predict(data[-15:])

        comparison = pd.DataFrame(dict(prediction=predictions, actual=toy_story[-15:]))
        movies_num = len(data.columns)
        print(f"{'='*8} {movies_num:6} movies {'='*8}")
        print(f"score: {lr.score(data[-15:], toy_story[-15:])}")
        print(comparison)

    print(scores)
    plt.title("model errors")
    plt.plot(max_movie_ids, scores)
    plt.xlabel("max movie id")
    plt.ylabel("error")
    plt.xscale('log')
    plt.show()

    plt.title("model errors")
    plt.plot([len(movie_ids[movie_ids < num]) for num in max_movie_ids], scores)
    plt.xlabel("number of movies")
    plt.ylabel("error")
    plt.xscale('log')
    plt.show()

    fig, ax = plt.subplots()
    ax.grid(True)
    xs = np.arange(len(toy_story[-15:])) + 1
    ax.scatter(xs, predictions, c='coral', s=50, label='predicted')
    ax.scatter(xs, toy_story[-15:], c='green', s=30, label='expected')
    ax.set_title(f'Regression model prediction results ({movies_num} movies)')
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
