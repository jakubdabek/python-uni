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


def main(dataset='ml-latest-small', ratings_limit=None):
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

    my_ratings2 = pd.DataFrame(dict(rating=my_ratings))
    my_ratings3 = sorted(list(my_ratings.items()), key=lambda x: x[0])
    my_ratings3 = sparse.csc_matrix(
        (
            [x for x, _ in my_ratings3],
            (np.zeros(len(my_ratings3)), [y for _, y in my_ratings3])
        ),
        (1, 10_001),
    )
    my_ratings = pd.DataFrame(dict(rating=my_ratings), index=movies.index)
    my_ratings = my_ratings.fillna(0.0)

    profile_timer, profile_iter_completed = Timer.make_iter_timer(
        100,
        callback=Timer.DEFAULT_CALLBACK,
    )

    def group_similarity(gr: pd.DataFrame) -> float:
        with contextlib.nullcontext():  # profile_iter_completed.manager():
            gr = gr.set_index('movieId')
            gr = gr.reindex(movies.index, fill_value=0.0)
            return cosine_similarity(gr.T, my_ratings.T)[0, 0]

    def group_similarity2(gr: pd.DataFrame) -> float:
        with contextlib.nullcontext():  # profile_iter_completed.manager():
            gr = gr.set_index('movieId')
            joined = gr.join(my_ratings2, rsuffix='_right', how='inner')
            if joined.empty:
                return 0.0
            left = joined["rating"].T.to_numpy().reshape(1, -1)
            right = joined["rating_right"].T.to_numpy().reshape(1, -1)
            return cosine_similarity(left, right)[0, 0]

    def group_similarity3(gr: pd.DataFrame) -> float:
        with contextlib.nullcontext():  # profile_iter_completed.manager():
            gr = gr.set_index('movieId')
            left = sparse.csc_matrix((gr["rating"], (np.zeros_like(gr.index), gr.index)), (1, my_ratings3.shape[1]))
            return cosine_similarity(left, my_ratings3)[0, 0]

    profile: ddf.DataFrame = ratings.groupby("userId").apply(group_similarity, meta=("cos", float))
    profile2: ddf.DataFrame = ratings.groupby("userId").apply(group_similarity2, meta=("cos", float))
    profile3: ddf.DataFrame = ratings.groupby("userId").apply(group_similarity3, meta=("cos", float))

    print("computing profile")
    with Timer(callback=Timer.DEFAULT_CALLBACK):
        profile: pd.Series = profile.compute()

    print("computing profile2")
    with Timer(callback=Timer.DEFAULT_CALLBACK):
        profile2: pd.Series = profile2.compute()

    print("computing profile3")
    with Timer(callback=Timer.DEFAULT_CALLBACK):
        profile3: pd.Series = profile3.compute()

    parquet_path = dataset_dir(dataset) / 'ratings_by_movie.parquet'

    print("saving intermediate ratings to parquet")
    with Timer(callback=Timer.DEFAULT_CALLBACK):
        ratings_by_movie = ratings \
            .reset_index() \
            .set_index('movieId', divisions=list(range(1, 10_001, 10))) \
            .to_parquet(parquet_path, engine="fastparquet")
        # .to_hdf(dataset_dir(dataset) / 'ratings_by_movie.hdf', "/data-*")

    from time import sleep
    # sleep(2)
    # ratings_by_movie = ddf.read_hdf(ratings_by_movie, "/data-*")
    ratings_by_movie: ddf.DataFrame = ddf.read_parquet(parquet_path)

    recommendations_timer, recommendations_iter_completed = Timer.make_iter_timer(
        100,
        callback=Timer.DEFAULT_CALLBACK,
    )

    def profile_similarity(gr: pd.DataFrame) -> float:
        with recommendations_iter_completed.manager():
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
        # recommendations15: pd.Series = recommendations.nlargest(15).compute()
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
