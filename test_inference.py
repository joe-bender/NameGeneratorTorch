import pytest
import torch
from inference import *

def test_pred():
    tests = [
        (torch.tensor([[[.2, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1]]]), 'a'),
        (torch.tensor([[[.1, .101, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1]]]), 'b'),
        (torch.tensor([[[.1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .11, .1]]]), 'z'),
        (torch.tensor([[[.1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, 2]]]), '_'),
    ]
    for input, output in tests:
        assert(pred_to_letter_det(input) == output)
        assert(pred_to_letter_rand(input) in string.ascii_lowercase+'_')

    tests = [0, 1, False, True, None, 'A', 'aa', '9', [1, 2], (1, 2), ['a'], ('a'), {3: 2},
        torch.randn(1, 1, 28),
        torch.randn(1, 2, 27),
        torch.randn(2, 1, 27),
    ]
    with pytest.raises(Exception):
        for input in tests:
            pred_to_letter_det(input)
            pred_to_letter_rand(input)
