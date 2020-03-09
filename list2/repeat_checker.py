import argparse
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Dict, List


def hash_file(file: Path) -> Tuple[int, str]:
    sha256 = hashlib.sha256()
    size = 0
    with file.open('rb') as f:
        for block in iter(lambda: f.read(sha256.block_size), b''):
            size += len(block)
            sha256.update(block)
    return size, sha256.hexdigest()


def find_duplicate_files(root: Path) -> Dict[Tuple[int, str], List[Path]]:
    contents: Dict[Tuple[int, str], List[Path]] = defaultdict(list)
    for file in root.rglob('*'):
        if file.is_file():
            contents[hash_file(file)].append(file)
    return contents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dirname', type=Path)
    root = parser.parse_args().dirname

    duplicates = find_duplicate_files(root)
    for same in duplicates.values():
        print('-' * 60)
        for file in same:
            print(file)


if __name__ == '__main__':
    main()
