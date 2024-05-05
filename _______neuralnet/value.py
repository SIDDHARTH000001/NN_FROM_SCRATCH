import math

class value:
  def __init__(self, data, _children = (), op = ""):
    self.data = data
    self._prev = set(_children)
    self._op = op
    self.grad = 0.0
    self._backward = lambda : None

  def __repr__(self):
    return f"value(data={self.data};_prev={self._prev},_op={self._op})"


  def __add__(self,other):
    other = other if isinstance(other,value) else value(other)
    out = value(self.data + other.data , (self , other) , '+')

    def _backward():
      self.grad +=  out.grad     # each variable might contribute to different nueron , which will lead gradient from differet out
      other.grad += out.grad
    out._backward = _backward

    return out

  def __radd__(self, other):
    return self.__add__(other)


  def __mul__(self,other):
    other = other if isinstance(other,value) else value(other)
    out = value(self.data * other.data , (self , other) , '*')

    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward

    return out
  
  def relu(self):
    out = value(0 if self.data < 0 else self.data, (self,), 'ReLU')
    
    def _backward():
        self.grad += (out.data > 0) * out.grad
    out._backward = _backward

    return out

  def __rmul__(self,other):
    return self.__mul__(other)


  def __sub__(self, other):
      return self + (-other)


  def __rsub__(self,other):
    return self.__sub__(other)

  def __truediv__(self, other):
    return self * other**-1


  def __pow__(self,other):
    assert isinstance(other,(int,float)), "only supporting int/float power"
    out = value(self.data ** other,(self,),"**")

    def _backward():
      self.grad += (other * ( self.data ** (other -1)) * out.grad)

    out._backward = _backward

    return out


  def tanh(self):
    x = self.data
    t = (math.exp( 2 * x) -1 ) / (math.exp(2 * x) +1)
    out = value(t , (self, ) , 'tahn')

    def _backward():
      self.grad += ( 1 - t ** 2 ) * out.grad
    out._backward = _backward

    return out

  def __neg__(self):
      return self * -1

  def __radd__(self, other): # other + self
        return self + other

  def __rsub__(self, other): # other - self
      return other + (-self)

  def __rtruediv__(self, other): # other / self
      return other * self**-1

  def backward(self):
    topo = []
    visited = set()

    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    build_topo(self)


    self.grad = 1            # for root node gradient will always be 1 
    for v in reversed(topo):
        v._backward()

