
class DualNumber:
    def __init__(self, a: float, b: float):
        '''
        Dual number of form a + b\epsilon
        '''
        self.a = a
        self.b = b
    
    def __inverse(self, other: DualNumber):
        # Inverse of the dual number
        return DualNumber(1/self.a, -self.b/(self.a * self.a))
    
    def __add__(self, other: DualNumber):
        return DualNumber(self.a + other.a, self.b + other.b)
    
    def __div__(self, other: DualNumber):
        return self / other.__inverse()
    
    def __mul__(self, other: DualNumber):
        return DualNumber(self.a * other.a, self.b * other.a + self.a * other.b)
    
    def __neg__(self):
        return DualNumber(-self.a, -self.b)
    
    def __sub__(self, other: DualNumber):
        return DualNumber(self.a - other.a, self.b - other.b)
    

