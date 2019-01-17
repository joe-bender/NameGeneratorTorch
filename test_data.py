from data import *
import pytest
import string

def test_get_names():
    names = get_names('names2017.csv')
    assert(type(names) is list)
    for name in names:
        assert(type(name) is str)
        for letter in name:
            assert(letter in string.ascii_letters)
        assert(len(names) == 32469)
