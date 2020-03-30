from __future__ import annotations

from numpy import log


class DualNumber:
    def __init__(self, a: float = 0., b: float = 0.):
        '''
        Dual number of form a + b\epsilon
        '''
        self.a = a
        self.b = b

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
        return DualNumber(self.a * other, self.b)

    __rmul__ = __mul__

    def __neg__(self):
        return DualNumber(-self.a, -self.b)

    def __pow__(self, other):
        if isinstance(other, DualNumber):
            new_a = self.a ** other.a
            new_b = new_a * (other.b * log(self.a) + self.b * other.a / self.a)
            return DualNumber(new_a, new_b)
        true_pow = self.a ** other
        return DualNumber(true_pow, true_pow * self.b * other / self.a)

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
