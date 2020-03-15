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
