from typing import Callable, List
from collections import deque
import heapq
import numpy as np

from .dualnumber import DualNumber


class CompGraphNode:
    def __init__(self, val, f = None, children = (), level = 0, in_path = False):
        self.val = val
        self.f = f
        self.children = children
        self.level = 0
        self.deriv = 0
        self.in_path = in_path
    
    @staticmethod
    def make_unary_parent(node, f):
        return CompGraphNode(f(node.val), f, (node,), node.level + 1, node.in_path)
    
    @staticmethod
    def make_binary_parent(node, other, f):
        max_level = max(node.level, other.level) + 1
        in_path = node.in_path or other.in_path
        return CompGraphNode(f(node.val, other.val), f, (node, other), max_level, in_path)

    @staticmethod
    def make_binary_parent_general(node, other, f):
        # Checks first if other is a CompGraphNode
        if isinstance(other, CompGraphNode):
            return CompGraphNode.make_binary_parent(node, other, f)
        return CompGraphNode.make_binary_parent(node, CompGraphNode(other), f)

    def sin(self):
        return CompGraphNode.make_unary_parent(self, np.sin)
    
    def cos(self):
        return CompGraphNode.make_unary_parent(self, np.cos)
    
    def tan(self):
        return CompGraphNode.make_unary_parent(self, np.tan)

    def __add__(self, other):
        return CompGraphNode.make_binary_parent_general(self, other, np.add)
    
    __radd__ = __add__

    def __div__(self, other):
        return CompGraphNode.make_binary_parent_general(self, other, np.divide)

    def __rdiv__(self, other):
        return CompGraphNode.make_binary_parent(CompGraphNode(other), self, np.divide)

    def __mul__(self, other):
        return CompGraphNode.make_binary_parent_general(self, other, np.multiply)

    __rmul__ = __mul__

    def __neg__(self):
        return CompGraphNode.make_unary_parent(self, np.negative)

    def __pow__(self, other):
        return CompGraphNode.make_binary_parent_general(self, other, np.power)

    def __rpow__(self, other):
        return CompGraphNode.make_binary_parent(CompGraphNode(other), self, np.power)

    def __sub__(self, other):
        return CompGraphNode.make_binary_parent_general(self, other, np.subtract)

    def __rsub__(self, other):
        return CompGraphNode.make_binary_parent(CompGraphNode(other), self, np.subtract)
    
    def __eq__(self, other):
        if isinstance(other, CompGraphNode):
            return self.val == other.val
        return self.val == other

    def __ne__(self, other):
        if isinstance(other, CompGraphNode):
            return self.val != other.val
        return self.val != other

    def __lt__(self, other):
        if isinstance(other, CompGraphNode):
            return self.val < other.val
        return self.val < other

    def __le__(self, other):
        if isinstance(other, CompGraphNode):
            return self.val <= other.val
        return self.val <= other

    def __gt__(self, other):
        if isinstance(other, CompGraphNode):
            return self.val > other.val
        return self.val > other

    def __ge__(self, other):
        if isinstance(other, CompGraphNode):
            return self.val >= other.val
        return self.val >= other


def grad(f: Callable) -> Callable:
    '''
    This function implements reverse mode autodifferentiation
    '''
    def forward_pass(args):
        arg_nodes = [CompGraphNode(v, in_path = True) for v in args]
        root_node = f(*arg_nodes)
        return root_node, arg_nodes
    
    def backward_pass(root_node: CompGraphNode, arg_nodes: List[CompGraphNode]):
        root_node.deriv = 1
        heap = [(-root_node.level, root_node)]
        while len(heap) > 0:
            level, first = heapq.heappop(heap)
            if len(first.children) == 0:
                continue
            elif len(first.children) == 1:
                child = first.children[0]
                child.deriv += first.deriv * first.f(DualNumber(child.val, 1)).b
            else: #2 children
                c1, c2 = first.children
                if c1.in_path:
                    c1.deriv += first.deriv * first.f(DualNumber(c1.val, 1), c2.val).b
                if c2.in_path:
                    c2.deriv += first.deriv * first.f(c1.val, DualNumber(c2.val, 1)).b
            for child in first.children:
                if child.in_path:
                    heapq.heappush(heap, (-child.level, child))
        return np.array(tuple(n.deriv for n in arg_nodes))
    
    def f_prime(*args) -> np.array:
        root_node, arg_nodes = forward_pass(args)
        if not isinstance(root_node, CompGraphNode):
            return np.array([0 for _ in args]) # Function output is not dependent on inputs
        g = backward_pass(root_node, arg_nodes)
        return g
    
    return f_prime