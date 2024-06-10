import numpy as np
import math

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)  # to save the information about the value's children (who create him, maybe parents is a better name)
        self._op = _op  # the operation to create this value, like '+' or '*'
        self.label = label
    
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, _children=(self, other), _op='+')
        def _backward():
            self.grad += 1* out.grad 
            other.grad += 1* out.grad 
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other 

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _children=(self, other), _op='*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self*other
    
    def tanh(self):
        x = self.data
        tanh = (math.exp(2*x) - 1)/(math.exp(2*x) + 1 )
        out = Value(tanh, (self, ), 'tanh')
        def _backward():
            self.grad += (1 - tanh**2) * out.grad 
        out._backward = _backward
        return out 
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), 'only supporting the int/float powers for now' 
        x = self.data
        out = Value(math.pow(x, other), (self, ), f'**{other}')
        def _backward():
            self.grad = other* (x**(other - 1)) * out.grad
        out._backward = _backward
        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):  # self / other
        return self * other ** -1

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # self - other
        return other + (-self)

    def __neg__(self):  # -self
        return self * -1

    def backward(self):
        self.grad = 1.0
        topo = []
        visited = set() 
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                # print (f'value: {v}')
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        for node in reversed(topo):
            node._backward() 