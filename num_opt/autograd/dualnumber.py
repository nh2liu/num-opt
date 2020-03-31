from __future__ import annotations

import numpy as np


class DualNumber:
    def __init__(self, a: float = 0., b: float = 0.):
        '''
        Dual number of form a + b\epsilon
        '''
        self.a = a
        self.b = b
    
    def sin(self):
        return DualNumber(np.sin(self.a), np.cos(self.a) * self.b)
    
    def cos(self):
        return DualNumber(np.cos(self.a), -np.sin(self.a) * self.b)
    
    def tan(self):
        return DualNumber(np.tan(self.a), 1/(np.cos(self.a) ** 2) * self.b)
  
    def __inverse(self):
        # Inverse of the dual number, unstable if a is very small
        return DualNumber(1 / self.a, -self.b / (self.a * self.a))

    def __add__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.a + other.a, self.b + other.b)
        return DualNumber(self.a + other, self.b)

    __radd__ = __add__

    def __div__(self, other):
        if isinstance(other, DualNumber):
            return self * other.__inverse()
        return DualNumber(self.a / other, self.b)

    def __rdiv__(self, other):
        return DualNumber(other, 0.) / self

    def __mul__(self, other):
        if isinstance(other, DualNumber):
            new_a = self.a * other.a
            new_b = self.b * other.a + self.a * other.b
            return DualNumber(new_a, new_b)
        return DualNumber(self.a * other, self.b * other)

    __rmul__ = __mul__

    def __neg__(self):
        return DualNumber(-self.a, -self.b)

    def __pow__(self, other):
        if isinstance(other, DualNumber):
            new_a = self.a ** other.a
            new_b = other.a * (self.a ** (other.a - 1)) * \
                b + np.log(self.a) * new_a * other.b
            return DualNumber(new_a, new_b)
        return DualNumber(self.a ** other,
                          other * (self.a ** (other - 1) * self.b))

    def __rpow__(self, other):
        return DualNumber(other, 0.) ** self

    def __repr__(self):
        return "D({:2f}, {:2f})".format(self.a, self.b)

    def __sub__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.a - other.a, self.b - other.b)
        return DualNumber(self.a - other, self.b)

    def __rsub__(self, other):
        return DualNumber(other, 0.) - self

    def __eq__(self, other):
        if isinstance(other, DualNumber):
            return (self.a, self.b) == (other.a, other.b)
        return self.a == other

    def __ne__(self, other):
        if isinstance(other, DualNumber):
            return (self.a, self.b) != (other.a, other.b)
        return self.a != other

    def __lt__(self, other):
        if isinstance(other, DualNumber):
            return ((self.a, self.b) < (other.a, other.b))
        return self.a < other

    def __le__(self, other):
        if isinstance(other, DualNumber):
            return ((self.a, self.b) <= (other.a, other.b))
        return self.a <= other

    def __gt__(self, other):
        if isinstance(other, DualNumber):
            return ((self.a, self.b) > (other.a, other.b))
        return self.a > other

    def __ge__(self, other):
        if isinstance(other, DualNumber):
            return ((self.a, self.b) >= (other.a, other.b))
        return self.a >= other