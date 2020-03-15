from typing import Callable


def logger(opt_func: Callable) -> Callable:
    '''
    Decorator around a optimization function that stores the input and output
    of every objective function call.
    '''
    x_hist = []
    y_hist = []

    def create_logged_func(f):
        def new_f(x):
            y = f(x)
            x_hist.append(x)
            y_hist.append(y)
            return y
        return new_f

    def ret_func(f, *args, **kwargs):
        new_f = create_logged_func(f)
        r = opt_func.py_func(new_f, *args, **kwargs)
        return r, x_hist, y_hist

    return ret_func
