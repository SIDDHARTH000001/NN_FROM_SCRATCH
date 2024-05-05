import random
from _______neuralnet import value


class Neuron:
  def __init__(self,N_in):
    self.w = [value(random.uniform(-1, 1)) for _ in range(N_in)]
    self.b = value(random.uniform(-1, 1))

  def __call__(self,x):
    acti = sum((wi * xi for wi,xi in zip(self.w, x)),self.b)
    out = acti.tanh()
    return out

  def parameters(self):
    return self.w + [self.b]


class Layer:
  def __init__(self, N_in, N_out):
    self.neurons = [Neuron(N_in) for _ in range(N_out)]

  def __call__(self,x):
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs)==1 else outs

  def parameters(self):
    return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
  def __init__(self, N_in , N_out):
    net = [N_in] + N_out
    self.layers = [Layer(net[i],net[i+1]) for i in range(len(N_out))]

  def __call__(self,x):
    for layer in self.layers:
      x = layer(x)
    return x

  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]
