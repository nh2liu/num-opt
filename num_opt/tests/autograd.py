import unittest
from .util import check_float, parabola, multivar, const_ret

from ..autograd.dualnumber import DualNumber
from ..autograd.forward import grad as fgrad
from ..autograd.reverse import grad as rgrad

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
        self.assertTrue(check_float(mult.b, 6.2))

        mult = 14 + DUAL_3
        self.assertTrue(check_float(mult.a, 16))
        self.assertTrue(check_float(mult.b, 6))

class ForwardAutoTest(unittest.TestCase):
    def test_basic(self):
        g = fgrad(const_ret.py_func)
        g_x, g_y, g_z = g(1, -4, -3)
        self.assertTrue(check_float(g_x, 0))
        self.assertTrue(check_float(g_y, 0))
        self.assertTrue(check_float(g_z, 0))
    
    def test_parabola(self):
        g = fgrad(parabola.py_func)
        self.assertTrue(check_float(g(1), -18))
        self.assertTrue(check_float(g(10), 0))
        self.assertTrue(check_float(g(11.3), 2.6))
    
    def test_multi(self):
        g = fgrad(multivar.py_func)
        g_x, g_y = g(0, 1)
        self.assertTrue(check_float(g_x, 11))
        self.assertTrue(check_float(g_y, 1))
        g_x, g_y = g(-1.3, 4.2)
        self.assertTrue(check_float(g_x, 11.6))
        self.assertTrue(check_float(g_y, -0.3))

class ReverseAutoTest(unittest.TestCase):
    def test_basic(self):
        g = rgrad(const_ret.py_func)
        g_x, g_y, g_z = g(1, -4, -3)
        self.assertTrue(check_float(g_x, 0))
        self.assertTrue(check_float(g_y, 0))
        self.assertTrue(check_float(g_z, 0))
    
    def test_parabola(self):
        g = rgrad(parabola.py_func)
        self.assertTrue(check_float(g(1), -18))
        self.assertTrue(check_float(g(10), 0))
        self.assertTrue(check_float(g(11.3), 2.6))
    
    def test_multi(self):
        g = rgrad(multivar.py_func)
        g_x, g_y = g(0, 1)
        self.assertTrue(check_float(g_x, 11))
        self.assertTrue(check_float(g_y, 1))
        g_x, g_y = g(-1.3, 4.2)
        self.assertTrue(check_float(g_x, 11.6))
        self.assertTrue(check_float(g_y, -0.3))