from numba import njit

# Helpful Test Functions
@njit
def unit_poly(x):
    return x*(x-1)*(x+2.5)*(x-3)

UNIT_POLY_MIN_X_1 = -1.661
UNIT_POLY_MIN_X_2 = 2.294

@njit
def parabola(x):
    return (x - 10) **2

PARABOLA_MIN = 10

# Actual test helpers.
def check_float(a: float, b: float, tol: float = 0.00001):
    if abs(a - b) <= tol:
        return True
    else:
        print(f"{a}, {b} do not fall within {tol}.")
        return False