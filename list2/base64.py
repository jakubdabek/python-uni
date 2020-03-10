import argparse
from typing import Tuple, Optional, Iterator, TypeVar, Iterable, BinaryIO, List

T = TypeVar('T')
IncompleteTriple = Tuple[T, Optional[T], Optional[T]]
Quad = Tuple[T, T, T, T]


base64_alphabet = b'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/='


def bytes_from_file(f: BinaryIO, chunk_size: int = 8192) -> Iterator[int]:
    while True:
        chunk = f.read(chunk_size)
        if chunk:
            for b in chunk:
                yield b
        else:
            break


def triples_padded(iterable: Iterable[T], padding: Optional[T] = None) -> Iterator[IncompleteTriple[T]]:
    result: List[Optional[T]] = [None] * 3
    i = 0
    for i, val in enumerate(iterable):
        result[i % 3] = val
        if i > 0 and i % 3 == 2:
            yield tuple(result)

    if i % 3 == 0:
        yield result[0], padding, padding
    elif i % 3 == 1:
        yield result[0], result[1], padding


def quads_assert(iterable: Iterable[T]) -> Iterator[Quad[T]]:
    result: List[Optional[T]] = [None] * 4
    i = 0
    for i, val in enumerate(iterable):
        result[i % 4] = val
        if i > 0 and i % 4 == 3:
            yield tuple(result)
    assert i % 4 == 3


def encode_single(v: int) -> int:
    return base64_alphabet[v]


def encode_triple(triple: IncompleteTriple[int]) -> Quad[int]:
    print([bin(x) for x in triple if x is not None])
    x1 = triple[0] >> 2
    x2 = (triple[0] << 4) & 0b_0011_0000
    if triple[1] is None:
        return tuple(map(encode_single, (x1, x2, -1, -1)))
    x2 |= triple[1] >> 4
    x3 = (triple[1] & 0b_0000_1111) << 2
    if triple[2] is None:
        return tuple(map(encode_single, (x1, x2, x3, -1)))
    x3 |= triple[2] >> 6
    x4 = triple[2] & 0b_0011_1111
    return tuple(map(encode_single, (x1, x2, x3, x4)))


def decode_single(v: int) -> int:
    return base64_alphabet.index(v)


def decode_quad(quad: Quad[int]) -> IncompleteTriple[int]:
    print(bytes(quad))
    x1 = decode_single(quad[0]) << 2
    q1 = decode_single(quad[1])
    x1 |= q1 >> 4
    x2 = (q1 & 0b_0000_1111) << 4
    if quad[2] == ord('='):
        if quad[3] != ord('='):
            raise ValueError("invalid base64 string")
        return x1, x2, None
    q2 = decode_single(quad[2])
    x2 |= q2 >> 2
    if quad[3] == ord('='):
        if q2 & 0b11 != 0:
            raise ValueError("invalid base64 string")
        return x1, x2, None
    x3 = (q2 & 0b11) << 6
    x3 |= decode_single(quad[3])
    return x1, x2, x3


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--decode", action='store_true', dest='decode')
    group.add_argument("--encode", action="store_false", dest='decode')
    parser.add_argument("input")
    parser.add_argument("output")

    args = parser.parse_args()
    if args.decode:
        with open(args.input, 'rb') as in_file, open(args.output, 'wb') as out_file:
            for quad in quads_assert(bytes_from_file(in_file)):
                decoded = decode_quad(quad)
                length = sum(1 for x in decoded if x is not None)
                print(decoded)
                out_file.write(bytes(decoded[:length]))
    else:
        with open(args.input, 'rb') as in_file, open(args.output, 'wb') as out_file:
            for triple in triples_padded(bytes_from_file(in_file)):
                out_file.write(bytes(b for b in encode_triple(triple) if b is not None))


if __name__ == '__main__':
    main()
