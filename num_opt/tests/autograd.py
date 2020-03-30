import unittest
from .util import check_float, parabola

from ..autograd.dualnumber import DualNumber
from ..autograd.forward import grad as fgrad

DUAL_1 = DualNumber(1.0, 2.0)
DUAL_2 = DualNumber(-3.1, 4.0)
DUAL_3 = DualNumber(2., 6.)

class DualNumberTest(unittest.TestCase):
    def test_add_sub(self):
        add = DUAL_1 + DUAL_2
        self.assertTrue(check_float(add.a, -2.1))
        self.assertTrue(check_float(add.b, 6))

        sub = DUAL_1 - DUAL_2
        self.assertTrue(check_float(sub.a, 4.1))
        self.assertTrue(check_float(sub.b, -2.0))

    def test_mult(self):
        mult = DUAL_1 * DUAL_2
        self.assertTrue(check_float(mult.a, -3.1))
        self.assertTrue(check_float(mult.b, -2.2))
        mult = DUAL_2 * DUAL_3
        self.assertTrue(check_float(mult.a, -6.2))
        self.assertTrue(check_float(mult.b, -10.6))
        
        inverse_check = DUAL_3 * DUAL_3._DualNumber__inverse()
        self.assertTrue(check_float(inverse_check.a, 1))
        self.assertTrue(check_float(inverse_check.b, 0))
    
    def test_float(self):
        mult = 3.1 * DUAL_1
        self.assertTrue(check_float(mult.a, 3.1))
        self.assertTrue(check_float(mult.b, 2.0))

        mult = 14 + DUAL_3
        self.assertTrue(check_float(mult.a, 16))
        self.assertTrue(check_float(mult.b, 6))

class ForwardAutoTest(unittest.TestCase):
    def test_parabola(self):
        self.parabola_grad = fgrad(parabola.py_func)
        self.assertTrue(check_float(self.parabola_grad(1), -18))
        self.assertTrue(check_float(self.parabola_grad(10), 0))
        self.assertTrue(check_float(self.parabola_grad(11), 2))