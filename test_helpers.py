import pytest
import torch
from helpers import *

def test_letter_to_onehot():
    tests = [
        ('a', torch.FloatTensor([[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]])),
        ('k', torch.FloatTensor([[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]])),
        ('z', torch.FloatTensor([[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]])),
        ('_', torch.FloatTensor([[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]])),
    ]
    for input, output in tests:
        assert torch.equal(letter_to_onehot(input), output)

    tests = [0, 1, False, True, None, 'A', 'aa', '9', [1, 2], (1, 2), ['a'], ('a'), {3: 2}]
    with pytest.raises(Exception):
        for input in tests:
            letter_to_onehot(input)

def test_letter_to_category():
    tests = [('a', 0), ('j', 9), ('z', 25), ('_', 26)]
    for input, output in tests:
        assert letter_to_category(input) == output

    tests = [0, 1, False, True, None, 'A', 'aa', '9', [1, 2], (1, 2), ['a'], ('a'), {3: 2}]
    with pytest.raises(Exception):
        for input in tests:
            letter_to_category(input)

def test_category_to_letter():
    tests = [(0, 'a'), (9, 'j'), (25, 'z'), (26, '_')]
    for input, output in tests:
        assert category_to_letter(input) == output

    tests = [-1, 27, 1.2, False, True, None, 'a', '9', [1, 2], (1, 2), (1), {3: 2}]
    with pytest.raises(Exception):
        for input in tests:
            category_to_letter(input)

def test_name_to_xy():
    tests = ['Abbie', 'Klaus', 'Zott', 'Cat', 'Jo', 'A']
    for input in tests:
        xs, ys = name_to_xy(input)
        assert(len(xs) == len(ys))
        for x, y in zip(xs, ys):
            assert(xs[0].size() == torch.Size([1, 1, 27]))
            assert(ys[0].size() == torch.Size([1]))

    tests = [False, True, None, 1, 1.2, [1, 2], (1, 3), {3: 2}, 'Andy_', 'Bob2', '3Joe']
    with pytest.raises(Exception):
        for input in tests:
            test_name_to_xy(input)
