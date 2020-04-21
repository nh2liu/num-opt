import numpy as np
from typing import List
from . import Npw

class Optimizer:
    '''
    Optimizer class that upgrades the parameters.
    '''
    def __init__(self, paramlist: List[Npw]):
        self.paramlist = paramlist
    
    def step(self):
        pass

class SGD(Optimizer):
    '''
    Implements stochastic gradient descent.
    '''
    def __init__(self, paramlist, lr = 0.1):
        super().__init__(paramlist)
        self.lr = lr
    
    def step(self):
        for npw in self.paramlist:
            npw.obj = npw.obj - self.lr * npw.grad

class Adam(Optimizer):
    '''
    Implements adam optimization.
    '''
    def __init__(self, paramlist, lr = 0.001, epsilon = 1e-08, b1 = 0.9, b2 = 0.999):
        super().__init__(paramlist)
        self.lr = lr
        self.epsilon = epsilon
        self.b1 = b1
        self.b2 = b2
        self.t = 1
        self.m = [0] * len(paramlist)
        self.v = [0] * len(paramlist)
    
    def step(self):
        for i, npw in enumerate(self.paramlist):
            self.m[i] = self.b1 * self.m[i] + (1-self.b1) * npw.grad
            self.v[i] = self.b2 * self.v[i] + (1-self.b2) * npw.grad * npw.grad
            
            mhat = self.m[i] / (1 - np.power(self.b1, self.t))
            vhat = self.v[i] / (1 - np.power(self.b2, self.t))
            npw.obj = npw.obj - self.lr * (mhat / (np.sqrt(vhat) + self.epsilon))
        self.t += 1