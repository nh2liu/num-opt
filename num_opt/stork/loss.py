import numpy as np
from . import Npw

class Loss:
    '''
    Class of functions that calculate the difference between
    a prediction and target.
    '''
    def __init__(self):
        pass

    def __call__(self, pred: Npw, actual: np.array) -> Npw:
        pass

class MSELoss(Loss):
    '''
    Calculated as `1/2 * * 1/ m * |pred - actual|_2`
    '''
    def __init__(self):
        super().__init__()
    
    def __call__(self, pred: Npw, actual: np.array):
        diff = pred - actual
        ret = np.array(np.mean(diff * diff)) 
        grad_fn = lambda next_grad: diff
        return Npw(ret, grad_fn, 'mse_loss', [pred])

class CrossEntropyLoss(Loss):
    '''
    Calculates the cross entropy loss:

    L = \sum_i^m y_i * \log (pred), y_i = 1 if i == actual else 0.
    '''
    def __init__(self):
        super().__init__()
        
    def __call__(self, pred: Npw, actual: np.array):
        m, n = pred.shape
        associated_output = np.take(pred, actual + n * np.arange(m))
        ret = np.array(np.mean(-np.log(associated_output)))
        d = np.zeros(pred.shape)
        
        for i in range(m):
            d[i, actual[i]] = -1 / associated_output[i]
        
        def grad_fn(next_grad):
            return d
        
        return Npw(ret, grad_fn, 'cross_entropy_loss', [pred])