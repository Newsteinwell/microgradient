# let's build a MLP with Neuron shows above
import numpy as np
import math

from micrograd_engine import Value

class Neuron():
    def __init__(self, nin, num_neuron=0, num_layer=0):
        self.nin = nin
        self.num_neuron = num_neuron
        self.num_layer=num_layer
        self.ws =[Value(np.random.normal(size=1)[0], label=f'w{i}{num_neuron}{num_layer}') for i in range(nin)] 
        # label i,j,k. i:id of input, j: id of output, k: number of layers
        self.b = Value(np.random.normal(size=1)[0], label=f'b{num_neuron}{num_layer}')
    
    def __call__(self, xin: list):
        self.a = sum([xi*wi for xi, wi in zip(xin, self.ws)], self.b)
        self.a.label = f'act{self.num_neuron}{self.num_layer}'
        # print (f'self.a: {self.a}')
        self.o = self.a.tanh()
        self.o.label = f'out{self.num_neuron}{self.num_layer}'
        return self.o
    
    def parameters(self):
        params = self.ws + [self.b]
        return params

class NNLayer():
    def __init__(self, nin, nout, num_layer=0):
        self.nin = nin
        self.nout = nout
        self.neurons = [Neuron(nin, num_neuron=i, num_layer=num_layer) for i in range(nout)]
    def __call__(self, xin):
        outs = [neuron(xin) for neuron in self.neurons]
        return outs
    
    def parameters(self):
        params = [] 
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params
    
class NNLayers():
    def __init__(self, nin, nouts: list):
        self.nin = nin
        self.nouts = nouts  # the number of neurons of each layer, a list
        num_neurons = [nin] + nouts
        self.layers = [NNLayer(num_neurons[i], num_neurons[i+1], num_layer=i+1) for i in range(len(num_neurons)-1)]
    
    def __call__(self, xin):
        for i in range(len(self.layers)):
            if i == 0:
                out = self.layers[0](xin)
            else:
                out = self.layers[i](out)
        return out
    
    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params