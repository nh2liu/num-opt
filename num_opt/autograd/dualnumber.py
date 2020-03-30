from __future__ import annotations

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

    def __add__(self, other: DualNumber):
        return DualNumber(self.a + other.a, self.b + other.b)

    def __div__(self, other: DualNumber):
        return self * other.__inverse()

    def __mul__(self, other: DualNumber):
        new_a = self.a * other.a
        new_b = self.b * other.a + self.a * other.b
        return DualNumber(new_a, new_b)

    def __neg__(self):
        return DualNumber(-self.a, -self.b)

    def __repr__(self):
        return "D({:2f}, {:2f})".format(self.a, self.b)

    def __sub__(self, other: DualNumber):
        return DualNumber(self.a - other.a, self.b - other.b)
