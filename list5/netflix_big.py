import contextlib
from pathlib import Path
from typing import Optional, Union, List

from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

import dask.dataframe as ddf
import pandas as pd
import numpy as np

from list4.time_decorator import Timer


def dataset_dir(dataset: str, parent_path: Optional[Union[Path, str]] = None) -> Path:
    if parent_path is None:
        parent_path = Path(__file__).parent
    elif not isinstance(parent_path, Path):
        parent_path = Path(parent_path)

    return parent_path / dataset


def load_large_data(
        file_name: str,
        *,
        usecols: Union[List[str], List[int]],
        dataset: str,
        parent_path: Optional[Union[Path, str]] = None,
) -> ddf.DataFrame:
    return ddf.read_csv(dataset_dir(dataset, parent_path) / file_name, usecols=usecols, blocksize=128 * 1024 * 1024)


def iter_timer(period=100, debug_level=1):
    if debug_level >= 1:
        timer, iter_completed = Timer.make_iter_timer(
            period,
            callback=Timer.DEFAULT_CALLBACK,
        )

        iter_completed_manager = iter_completed.manager if debug_level >= 2 else contextlib.nullcontext
    else:
        timer, iter_completed_manager = contextlib.nullcontext(), contextlib.nullcontext

    return timer, iter_completed_manager


def main(dataset='ml-latest-small', ratings_limit=None, debug_level=1):
    movies: ddf.DataFrame = load_large_data("movies.csv", usecols=["movieId", "title"], dataset=dataset)
    movies = movies.set_index("movieId", sorted=True)
    movies: pd.DataFrame = movies.loc[:10_001].compute()
    ratings = load_large_data("ratings.csv", usecols=["userId", "movieId", "rating"], dataset=dataset)
    ratings = ratings[ratings["movieId"] <= 10_000]
    ratings: ddf.DataFrame = ratings.set_index("userId", sorted=True)
    ratings = ratings.repartition(divisions=ratings.divisions)
    if ratings_limit is not None:
        ratings = ratings.loc[:ratings_limit]

    my_ratings = {
        2571: 5.0,
        32: 4.0,
        260: 5.0,
        1097: 4.0,
    }

    my_ratings = sorted(list(my_ratings.items()), key=lambda x: x[0])
    my_ratings = sparse.csc_matrix(
        (
            [rating for _, rating in my_ratings],
            (np.zeros(len(my_ratings)), [pos for pos, _ in my_ratings])
        ),
        (1, 10_001),
    )

    profile_timer, profile_iter_completed_manager = iter_timer(debug_level=debug_level)

    def group_similarity(gr: pd.DataFrame) -> float:
        with profile_iter_completed_manager():
            gr = gr.set_index('movieId')
            left = sparse.csc_matrix((gr["rating"], (np.zeros_like(gr.index), gr.index)), (1, my_ratings.shape[1]))
            return cosine_similarity(left, my_ratings)[0, 0]

    profile: ddf.DataFrame = ratings.groupby("userId").apply(group_similarity, meta=("cos", float))

    print("computing profile")
    with profile_timer:
        profile: pd.Series = profile.compute()

    parquet_path = dataset_dir(dataset) / 'ratings_by_movie.parquet'

    print("saving intermediate ratings to parquet")
    with Timer(callback=Timer.DEFAULT_CALLBACK if debug_level >= 1 else None):
        ratings \
            .reset_index() \
            .set_index('movieId', divisions=list(range(1, 10_001, 100))) \
            .to_parquet(parquet_path, engine="fastparquet")

    ratings_by_movie: ddf.DataFrame = ddf.read_parquet(parquet_path)

    recommendations_timer, recommendations_iter_completed_manager = iter_timer(debug_level=debug_level)

    def profile_similarity(gr: pd.DataFrame) -> float:
        with recommendations_iter_completed_manager():
            gr = gr.set_index("userId")
            joined = gr.join(profile, how='outer').fillna(0.0)
            if joined.empty:
                return 0.0
            left = joined["rating"].T.to_numpy().reshape(1, -1)
            right = joined["cos"].T.to_numpy().reshape(1, -1)
            return cosine_similarity(left, right)[0, 0]

    recommendations: ddf.Series = ratings_by_movie \
        .groupby("movieId") \
        .apply(profile_similarity, meta=("recommendation", float))

    print("computing recommendations")
    with recommendations_timer:
        recommendations: pd.Series = recommendations.compute()

    recommendations: pd.DataFrame = movies.join(recommendations).fillna(0.0)
    recommendations.sort_values("recommendation", ascending=False, inplace=True)

    return movies, ratings, my_ratings, profile, ratings_by_movie, recommendations


def limit_memory(maxsize):
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (maxsize, hard))
    except ImportError:
        print("couldn't limit memory")


limit_memory(1024 * 1024 * 1024)

if __name__ == '__main__':
    main()
