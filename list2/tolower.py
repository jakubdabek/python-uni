import random
import shutil
import string
import sys
from pathlib import Path


alphabet = string.ascii_letters + string.digits
extensions = [".txt", ".bin", ".test"]


def generate_random_files(root: Path):
    def generate_dir(dirname: Path, depth=0):
        if dirname.exists():
            print("Removing", dirname)
            shutil.rmtree(dirname)

        print("Creating", dirname)
        dirname.mkdir()
        for _ in range(int(min(max(random.normalvariate(5, 4), 2), 10))):
            name = ''.join(random.choices(alphabet, k=random.randint(4, 12)))
            if depth < 4 and random.random() > 0.7:
                generate_dir(dirname / name, depth + 1)
            else:
                ext = random.choice(extensions) if random.random() > 0.6 else random.choice(extensions).upper()
                file = dirname / (name + ext)
                print("Writing", file)
                file.write_text(f"test{random.randint(1, 5)}")
    generate_dir(root)


def transform_files(root: Path, check_only=True):
    for file in root.rglob('*'):
        print(f"Renaming '{file}' -> '{file.with_name(file.name.lower())}")
        if not check_only:
            file.rename(Path(file.name.lower()))


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} (--generate | --check-only | --transform) <dirname>")
        sys.exit(1)

    root = Path(sys.argv[2])
    if sys.argv[1] == '--check-only':
        transform_files(root)
    elif sys.argv[1] == '--transform':
        transform_files(root, check_only=False)
    elif sys.argv[1] == '--generate':
        generate_random_files(root)


if __name__ == '__main__':
    main()
