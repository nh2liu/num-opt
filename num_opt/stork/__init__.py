from typing import List
import numpy as np

class Npw:
    '''
    Numpy wrapper class (equivalent of torch tensor).
    Can call backwards on it which calculates the gradients.
    '''
    def __init__(self, obj, grad_fn = None, grad_name = None, children = [], pchildren = []):
        assert(isinstance(obj, np.ndarray))
        self.obj = obj
        self.grad_fn = grad_fn
        self.children = children
        self.pchildren = pchildren
        self.grad_name = grad_name
    
    def backward(self, x = None):
        if self.grad_fn is not None:
            self.grad = self.grad_fn(x)
        
        for child in self.children:
            child.backward(self.grad)
        
        for child in self.pchildren:
            child.backward(x)
    
    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.obj, attr)
    
    def __repr__(self):
        s = self.obj.__repr__().replace('array', 'Npw')
        s = s.replace(')', f', grad = {self.grad_name})')
        return s

class Sequential:
    '''
    A sequential model. Example usage:
    
    model = Sequential(
        Linear(784, 150),
        Relu(),
        Linear(150, 10),
        Softmax(),
    )
    '''
    def __init__(self, *args: List):
        self.ops = args
        if len(args) == 0:
            raise Exception('Need at least 1 operator.')
        
        cur_out = self.ops[0].in_dim
        self._params = []
        for i, op in enumerate(self.ops):
            if op.in_dim != -1 and op.in_dim != cur_out:
                raise Exception(f'Input of op {i + 1}: {op}, does not match output of previous operator, {cur_out}.')
            if op.out_dim != -1:
                cur_out = op.out_dim
            self._params.extend(op.parameters())
    
    def parameters(self):
        return self._params
    
    def __call__(self, x):
        x = Npw(x)
        for op in self.ops:
            x = op(x)
        return x