from numba import njit

@njit
def unit_poly(x):
    return x*(x-1)*(x+2.5)*(x-3)

UNIT_POLY_MIN_X_1 = -1.661
UNIT_POLY_MIN_X_2 = 2.294

@njit
def parabola(x):
    return (x - 10) **2

PARABOLA_MIN = 10