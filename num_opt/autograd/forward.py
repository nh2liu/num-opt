from typing import Callable
from .dualnumber import DualNumber
import numpy as np


def grad(f: Callable) -> Callable:
    '''
    This function implements forward mode autodifferentiation
    '''
    def get_partial(i: int, args: list) -> float:
        # Runs f() once to calculate the ith gradient
        v = args[i]
        args[i] = DualNumber(v, 1.)
        partial = f(*args).b
        # Resetting the args
        args[i] = v
        return partial
    
    def f_prime(*args) -> np.array:
        args = list(args)
        g = np.empty(len(args))
        for i in range(len(args)):
            g[i] = get_partial(i, args)
        return g
    
    return f_prime