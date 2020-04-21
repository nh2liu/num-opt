import numpy as np
from . import Npw

class Op:
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
    
    def parameters(self):
        return []

class Linear(Op):
    def __init__(self, in_features, out_features, bias = True):
        super().__init__(in_features, out_features)
        self.bias = bias
        if self.bias:
            in_features += 1
        self.weights = Npw(np.random.normal(loc=0.0, scale= 0.1, size=(in_features, out_features)))
        self.weights.grad_name = 'matmul'
    
    # x.T @ next_layer.grad()
    def __call__(self, x):
        v = x.obj
        weights = self.weights.obj
        if self.bias:
            v = np.hstack((v, np.ones(v.shape[0]).reshape(-1,1)))
            weights = weights[:-1, :]
        
        def grad_fn(next_grad):
            return (np.transpose(v) @ next_grad) / v.shape[0]
        
        self.weights.grad_fn = grad_fn
        
        def lop_fn(next_grad):
            return next_grad @ np.transpose(weights)
        
        
        ret = v @ self.weights
        return Npw(ret, lop_fn, 'linear', [x], [self.weights])
    
    def __repr__(self):
        i, j = self.weights.shape
        return f'Linear({i}, {j}, bias = {self.bias})'
    
    def parameters(self):
        return [self.weights]
    
class Relu(Op):
    def __init__(self):
        super().__init__(-1, -1)
    
    def __call__(self, x):
        def grad_fn(next_grad):
            replace_0 = np.where(x.obj < 0, 0, x)
            return np.where(replace_0 > 0, 1., replace_0) * next_grad
        ret = np.maximum(x, 0)
        return Npw(ret, grad_fn, 'relu', [x])

class Leakyrelu(Op):
    def __init__(self, slope = 0.1):
        super().__init__(-1, -1)
        self.s = slope
    
    def __call__(self, x):
        def grad_fn(next_grad):
            replace_0 = np.where(x.obj < 0, self.s, x)
            return np.where(replace_0 > 0, 1., replace_0) * next_grad
        ret = np.maximum(x, 0) + self.s * np.minimum(x, 0)
        return Npw(ret, grad_fn, 'leaky-relu', [x])

class Sigmoid(Op):
    def __init__(self):
        super().__init__(-1, -1)
    
    def __call__(self, x):
        c = np.exp(x)
        ret = c / (c + 1)
        def grad_fn(next_grad):
            return ret * (1 - ret) * next_grad
        return Npw(ret, grad_fn, 'sigmoid', [x])

class Softmax(Op):
    def __init__(self):
        super().__init__(-1, -1)
    
    def __call__(self, x):
        m, n = x.shape
        c = np.exp(x)
        ret = c / (np.sum(c, axis = 1).reshape(-1, 1))
        
        def jacobi(a):
            a_ = a.reshape(1, -1)
            return np.diag(a) - a_.T @ a_
        
        jacobians = np.apply_along_axis(jacobi, 1, ret)
        def grad_fn(next_grad):
            return (next_grad.reshape(m, 1, n) @ jacobians).reshape(m, n)
        return Npw(ret, grad_fn, 'softmax', [x])
