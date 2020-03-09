import argparse
import random
from math import gcd


def prime_mod_inv(x, mod):
    return pow(x, mod - 2, mod)


def fermat(n, k) -> bool:
    """Performs the Fermat primality test and returns whether the number is definitely composite"""
    for _ in range(k):
        a = random.randint(2, n - 2)
        if pow(a, n - 1, n) != 1:
            return True
    return False


def miller_rabin(n, k) -> bool:
    """Performs the Miller-Rabin primality test and returns whether the number is definitely composite"""
    assert n > 3 and n % 2 == 1
    assert k > 0

    # calculate n = 2^r * d + 1
    r = 0
    d = n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    for _ in range(k):
        a = random.randint(2, n - 2)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                continue
        return True
    return False


def gen_keys(args):
    pass


def encrypt(args):
    pass


def decrypt(args):
    pass


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    gen_keys_parser = subparsers.add_parser('gen-keys')
    gen_keys_parser.add_argument('bits', type=int)
    gen_keys_parser.set_defaults(execute=gen_keys)
    encrypt_parser = subparsers.add_parser('encrypt')
    encrypt_parser.add_argument('text')
    encrypt_parser.set_defaults(execute=encrypt)
    decrypt_parser = subparsers.add_parser('decrypt')
    decrypt_parser.add_argument('text')
    decrypt_parser.set_defaults(execute=decrypt)

    args = parser.parse_args()
    args.execute(args)


if __name__ == '__main__':
    main()
