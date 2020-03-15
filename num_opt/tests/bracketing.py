import unittest
from ..algorithms.bracketing import *
from ..algorithms.exceptions import MaxIterationException
from .util import *

class TestBracketing(unittest.TestCase):
    def test_basic_constraint(self):
        a, b = bracket(unit_poly, x0 = 0)
        self.assertTrue(a < b)

        with self.assertRaises(MaxIterationException):
            bracket(unit_poly, x0 = 10, max_iter = 2)

    def test_bound_contains_min(self):
        # Testing if the bounds found contain a minimum
        a, b = bracket(unit_poly, x0 = -2)
        self.assertTrue(a < UNIT_POLY_MIN_X_1)
        self.assertTrue(b > UNIT_POLY_MIN_X_1)
        
        a, b = bracket(unit_poly, x0 = -1)
        self.assertTrue(a < UNIT_POLY_MIN_X_1)
        self.assertTrue(b > UNIT_POLY_MIN_X_1)

        a, b = bracket(unit_poly, x0 = 1)
        self.assertTrue(a < UNIT_POLY_MIN_X_2)
        self.assertTrue(b > UNIT_POLY_MIN_X_2)

        a, b = bracket(parabola, x0 = 50)
        self.assertTrue(a < PARABOLA_MIN)
        self.assertTrue(b > PARABOLA_MIN)

class TestFibonacci(unittest.TestCase):
    def test_basic_constraint(self):
        a, b = fibonacci_search(unit_poly, a = 1, b = 3, n = 2)
        self.assertTrue(a < b)

        # Test if new bounds are smaller with larger n
        a2, b2 = fibonacci_search(unit_poly, a = 1, b = 3, n = 4)
        self.assertTrue(a2 > a)
        self.assertTrue(b2 < b)

    def test_bound_contains_min(self):
        # Testing if the bounds found contain a minimum
        a, b = fibonacci_search(unit_poly, a = -2, b = 0, n = 7)
        self.assertTrue(a < UNIT_POLY_MIN_X_1)
        self.assertTrue(b > UNIT_POLY_MIN_X_1)
        
        a, b = fibonacci_search(unit_poly, a = 1, b = 3, n = 7)
        self.assertTrue(a < UNIT_POLY_MIN_X_2)
        self.assertTrue(b > UNIT_POLY_MIN_X_2)

        a, b = fibonacci_search(parabola, a = 1, b = 15, n = 5)
        self.assertTrue(a < PARABOLA_MIN)
        self.assertTrue(b > PARABOLA_MIN)