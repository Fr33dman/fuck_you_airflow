import random
import string

ALPHABET = string.ascii_letters


def random_string(length: int = 5):
    return ''.join(list(random.choice(ALPHABET) for i in range(length)))


def pairs(*args):
    pair_list = [None] + list(*args) + [None]
    length = len(pair_list)
    for idx in range(length - 1):
        yield pair_list[idx], pair_list[idx + 1]
