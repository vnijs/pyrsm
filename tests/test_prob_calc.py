import math
import numpy as np
from scipy import stats

from pyrsm.basics.prob_calc import binomial, poisson, normal


def test_binomial_basic():
    # compare pmf and cdf for a small example
    n = 10
    p = 0.3
    res = binomial.prob_binom(n=n, p=p, lb=2, ub=4)
    # manual checks
    assert res["n"] == n
    assert math.isclose(res["p_elb"], stats.binom.pmf(2, n, p), rel_tol=1e-9)
    assert math.isclose(res["p_eub"], stats.binom.pmf(4, n, p), rel_tol=1e-9)
    assert math.isclose(
        res["p_int"], sum(stats.binom.pmf(k, n, p) for k in range(2, 5)), rel_tol=1e-9
    )


def test_poisson_basic():
    lam = 2.5
    res = poisson.prob_pois(lamb=lam, lb=0, ub=3)
    assert res["lamb"] == lam
    # pmf at 0
    assert math.isclose(res["p_elb"], stats.poisson.pmf(0, lam), rel_tol=1e-9)
    assert math.isclose(res["p_eub"], stats.poisson.pmf(3, lam), rel_tol=1e-9)
    assert math.isclose(
        res["p_int"], sum(stats.poisson.pmf(k, lam) for k in range(0, 4)), rel_tol=1e-9
    )


def test_normal_basic():
    mean = 0
    stdev = 1
    res = normal.prob_norm(mean=mean, stdev=stdev, lb=-1, ub=1)
    assert math.isclose(res["p_lb"], stats.norm.cdf(-1, mean, stdev), rel_tol=1e-9)
    assert math.isclose(res["p_ub"], stats.norm.cdf(1, mean, stdev), rel_tol=1e-9)
    assert math.isclose(
        res["p_int"], stats.norm.cdf(1, mean, stdev) - stats.norm.cdf(-1, mean, stdev), rel_tol=1e-9
    )
