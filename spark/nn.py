import numpy as np
import cPickle as pickle
from classifier import Classifier
from util.layers import *
from util.dump import *

""" STEP2: Build Two-layer Fully-Connected Neural Network """

class NNClassifier(Classifier):
  def __init__(self, D, H, W, K, iternum):
    Classifier.__init__(self, D, H, W, K, iternum)
    self.L = 100 # size of hidden layer

    """ Layer 1 Parameters """
    # weight matrix: [M * L]
    self.A1 = 0.01 * np.random.randn(self.M, self.L)
    # bias: [1 * L]
    self.b1 = np.zeros((1,self.L))

    """ Layer 3 Parameters """
    # weight matrix: [L * K]
    self.A3 = 0.01 * np.random.randn(self.L, K)
    # bias: [1 * K]
    self.b3 = np.zeros((1,K))

    """ Hyperparams """
    # learning rate
    self.rho = 1e-2
    # momentum
    self.mu = 0.9
    # reg strencth
    self.lam = 0.1
    # velocity for A1: [M * L]
    self.v1 = np.zeros((self.M, self.L))
    # velocity for A3: [L * K] 
    self.v3 = np.zeros((self.L, K))
    return

  def load(self, path):
    data = pickle.load(open(path + "layer1"))
    assert(self.A1.shape == data['w'].shape)
    assert(self.b1.shape == data['b'].shape)
    self.A1 = data['w']
    self.b1 = data['b']
    data = pickle.load(open(path + "layer3"))
    assert(self.A3.shape == data['w'].shape)
    assert(self.b3.shape == data['b'].shape)
    self.A3 = data['w']
    self.b3 = data['b']
    return

  def param(self):
    return [("A1", self.A1), ("b1", self.b1), ("A3", self.A3), ("b3", self.b3)]

  def forward(self, data):
    """
    INPUT:
      - data: RDD[(key, (images, labels)) pairs]
    OUTPUT:
      - RDD[(key, (images, list of layers, labels)) pairs]
    """
    """
    TODO: Implement the forward passes of the following layers
    Layer 1 : linear
    Layer 2 : ReLU
    Layer 3 : linear
    """
    return data.map(lambda (k, (x, y)): (k, (x, [np.zeros((x.shape[0], 2))], y))) # replace it with your code

  def backward(self, data, count):
    """
    INPUT:
      - data: RDD[(images, list of layers, labels) pairs]
    OUTPUT:
      - loss
    """
    """ 
    TODO: Implement softmax loss layer 
    """

    """ Todo: Implement backpropagation for Layer 3 """

    """ Todo: Implement backpropagation for Layer 2 """
   
    """ Todo: Implmenet backpropagation for Layer 1 """

    """ Todo: Reduce gradients """
    L = 0.0
    dLdA3 = np.zeros(self.A3.shape)
    dLdb3 = np.zeros(self.b3.shape)
    dLdA1 = np.zeros(self.A1.shape)
    dLdb1 = np.zeros(self.b1.shape)

    """ gradient scaling """
    L /= float(count)
    dLdA3 /= float(count)
    dLdb3 /= float(count)
    dLdA1 /= float(count)
    dLdb1 /= float(count)
     
    """ regularization """
    L += 0.5 * self.lam * (np.sum(self.A1*self.A1) + np.sum(self.A3*self.A3))
 
    """ regularization gradient """
    dLdA3 = dLdA3.reshape(self.A3.shape)
    dLdA1 = dLdA1.reshape(self.A1.shape)
    dLdA3 += self.lam * self.A3
    dLdA1 += self.lam * self.A1

    """ tune the parameter """
    self.v1 = self.mu * self.v1 - self.rho * dLdA1
    self.v3 = self.mu * self.v3 - self.rho * dLdA3
    self.A1 += self.v1
    self.A3 += self.v3
    self.b1 += - self.rho * dLdb1
    self.b3 += - self.rho * dLdb3

    return L
