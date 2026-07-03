"""A tiny calculator module used by the smoke fixture repo."""


def add(a, b):
    return a + b


def sub(a, b):
    return a - b


def mul(a, b):
    # BUG: multiplication is implemented as repeated addition but the loop
    # bound is off by one, so mul(a, b) returns a * (b + 1) for b >= 0.
    total = 0
    for _ in range(b + 1):
        total = add(total, a)
    return total


def clamp(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x
