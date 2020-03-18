import argparse
import itertools
import math
import binascii
import random
from typing import Iterator


def prime_mod_inv(x, mod):
    return pow(x, mod - 2, mod)


def egcd(a: int, b: int) -> (int, int, int):
    """Extended Euclidean algorithm"""
    if a == 0:
        return b, 0, 1
    else:
        g, y, x = egcd(b % a, a)
        return g, x - (b // a) * y, y


def modinv(a: int, m: int) -> int:
    g, x, y = egcd(a, m)
    if g != 1:
        raise ValueError('modular inverse does not exist')
    else:
        return x % m


def fermat(n: int, k: int) -> bool:
    """Perform the Fermat primality test and return whether the number is definitely composite"""
    for _ in range(k):
        a = random.randint(2, n - 2)
        if pow(a, n - 1, n) != 1:
            return True
    return False


def miller_rabin(n: int, k: int) -> bool:
    """Perform the Miller-Rabin primality test and return whether the number is definitely composite"""
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


def is_very_probably_prime(num: int) -> bool:
    return not fermat(num, 64) and not miller_rabin(num, 64)


# checking exactly for primeness is too slow, so this is not used
def is_prime(num: int) -> bool:
    if num % 2 == 0:
        return False
    for divisor in itertools.count(3, 2):
        if divisor * divisor > num:
            break
        if num % divisor == 0:
            return False
    return True


def primes(lower: int, upper: int) -> Iterator[int]:
    while True:
        num = random.randint(lower, upper)
        if is_very_probably_prime(num):
            print("found prime", num)
            yield num


def str2i(message: bytes) -> int:
    return int(binascii.hexlify(message), 16)


def clean_hex(x: int) -> str:
    h = hex(x)[2:]
    if h[-1] == 'L':
        h = h[:-1]
    if len(h) & 1 == 1:
        h = '0' + h
    return h


def i2str(encoded_message: int) -> bytes:
    h = clean_hex(encoded_message)
    return binascii.unhexlify(h)


class Keys:
    def __init__(self, d: int, n: int, e: int):
        self.d, self.n, self.e = d, n, e

    @property
    def dne(self) -> (int, int, int):
        return self.d, self.n, self.e

    @property
    def private_key(self) -> int:
        return self.d

    @property
    def public_key(self) -> (int, int):
        return self.n, self.e

    def encrypt(self, message: bytes) -> int:
        m = str2i(message)
        return pow(m, self.e, self.n)

    def decrypt(self, encrypted_message: int) -> bytes:
        m = pow(encrypted_message, self.d, self.n)
        return i2str(m)

    def write_to_file(self, *, pub_filename: str, prv_filename: str):
        with open(pub_filename, 'wb') as pub_file, open(prv_filename, 'wb') as prv_file:
            d, n, e = self.dne
            d = clean_hex(d).encode('utf-8')
            n = clean_hex(n).encode('utf-8')
            e = clean_hex(e).encode('utf-8')

            pub_file.write(n)
            pub_file.write(b'\n')
            pub_file.write(e)
            pub_file.write(b'\n')

            prv_file.write(d)
            prv_file.write(b'\n')
            prv_file.write(e)
            prv_file.write(b'\n')

    @classmethod
    def from_bits(cls, bits: int):
        possible_ps = primes(2 ** (bits // 2), 2 ** (bits // 2 + 1) - 1)
        p = next(p for p in possible_ps if is_very_probably_prime((p - 1) // 2))
        print(f"found p: {p}")
        q_bounds = (2 ** (bits - 1)) // p, (2 ** bits - 1) // p
        print(f"bounds for q: {q_bounds}")
        possible_qs = primes(*q_bounds)
        # q = next(filter(functools.partial(operator.ne, p), possible_qs))
        q = next(q for q in possible_qs if q != p)

        n = p * q
        phi = (p - 1) * (q - 1)

        real_bits = math.log(n, 2)
        print(f"n: {n}, log_2(n): {real_bits}, phi: {phi}")
        assert math.ceil(real_bits) == bits

        e = 2 ** 16 + 1
        assert math.gcd(e, phi) == 1
        d = modinv(e, phi)

        return cls(d, n, e)

    @classmethod
    def from_file(cls, *, pub_filename: str, prv_filename: str):
        with open(pub_filename, 'rb') as pub_file, open(prv_filename, 'rb') as prv_file:
            n = int(pub_file.readline(), 16)
            e = int(pub_file.readline(), 16)
            d = int(prv_file.readline(), 16)
            e2 = int(prv_file.readline(), 16)
            assert e == e2

            print(f"d: {d}, n: {n}, e: {e}")
            return cls(d, n, e)


def gen_keys(args):
    keys = Keys.from_bits(args.bits)
    keys.write_to_file(pub_filename='key.pub', prv_filename='key.prv')


def encrypt(args):
    text: str = args.text
    keys = Keys.from_file(pub_filename='key.pub', prv_filename='key.prv')
    print(keys.encrypt(text.encode('utf-8')))


def decrypt(args):
    text: str = args.text
    keys = Keys.from_file(pub_filename='key.pub', prv_filename='key.prv')
    print(keys.decrypt(int(text)))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True, dest='command')
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
