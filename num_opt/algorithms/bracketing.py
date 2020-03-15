import sys

from typing import Callable, Tuple
import numpy as np
from numba import njit
from .exceptions import MaxIterationException


@njit
def bracket(
        f: Callable,
        x0: float = 0.,
        s: float = 0.001,
        k: float = 2.,
        max_iter: int = 100) -> Tuple[float, float]:
    '''
    Returns a range (a,b) that a local minima lies in.
    '''
    x1, x2 = x0, x0 + s
    y1, y2 = f(x0), f(x0 + s)
    if y2 > y1:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        s = -s
    for _ in range(max_iter):
        x3 = x2 + s
        y3 = f(x3)
        if y3 > y2:
            return (x1, x3) if x3 > x1 else (x3, x1)
        x1, x2 = x2, x3
        y1, y2 = y2, y3
        s *= k
    raise MaxIterationException


@njit
def fibonacci_search(
        f: Callable,
        a: float,
        b: float,
        n: int
    ) -> Tuple[float, float]:
    '''
    Fibonacci searches within the interval (a,b) to find a smaller interval.
    Querys function f a maximum of n times.
    '''
    p = (1 + np.sqrt(5)) / 2 - 1
    x2 = (1 - p) * a + p * b
    y2 = f(x2)
    for i in range(n - 1):
        x1 = p * a + (1 - p) * b
        y1 = f(x1)
        if y1 < y2:
            b = x2
            x2, y2 = x1, y1
        else:
            a, b = b, x1
    return (a, b) if a < b else (b, a)
