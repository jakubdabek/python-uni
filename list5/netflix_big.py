from pathlib import Path
from typing import Optional, Union, List

from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

import dask.dataframe as ddf
import pandas as pd
import numpy as np


def load_large_data(
        file_name: str,
        *,
        usecols: Union[List[str], List[int]],
        dataset: str,
        dir_path: Optional[Union[Path, str]] = None,
) -> ddf.DataFrame:
    if dir_path is None:
        dir_path = Path(__file__).parent / dataset
    elif not isinstance(dir_path, Path):
        dir_path = Path(dir_path)

    return ddf.read_csv(dir_path / file_name, usecols=usecols, blocksize=128 * 1024 * 1024)


def main(dataset='ml-latest-small'):
    movies: ddf.DataFrame = load_large_data("movies.csv", usecols=["movieId", "title"], dataset=dataset)
    movies = movies.set_index("movieId", sorted=True)
    movies: pd.DataFrame = movies.loc[:10_001].compute()
    ratings = load_large_data("ratings.csv", usecols=["userId", "movieId", "rating"], dataset=dataset)
    ratings: ddf.DataFrame = ratings.set_index("userId", sorted=True)
    ratings = ratings.repartition(divisions=ratings.divisions)

    my_ratings = {
        2571: 5.0,
        32: 4.0,
        260: 5.0,
        1097: 4.0,
    }
    my_ratings = pd.DataFrame(dict(rating=my_ratings), index=movies.index)
    my_ratings = my_ratings.fillna(0.0)

    def group_similarity(gr: pd.DataFrame) -> float:
        gr = gr.set_index('movieId')
        gr = gr.reindex(movies.index).fillna(0.0)
        return cosine_similarity(gr.T, my_ratings.T)[0, 0]

    profile = ratings.groupby("userId").apply(group_similarity, meta=float)

    return movies, ratings, my_ratings, profile


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
