"""String helpers used by the smoke fixture repo."""


def normalize(s):
    return s.strip().lower()


def is_palindrome(s):
    t = normalize(s)
    return t == t[::-1]
