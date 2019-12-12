from pyrsm.stats import varprop, seprop


def test_varprop():
    assert varprop([1, 1, 1, 0, 0, 0]) == 0.25, "Proportion standard error incorrect"


def test_seprop():
    assert (
        seprop([1, 1, 1, 0, 0, 0]) == 0.2041241452319315
    ), "Proportion standard error incorrect"
