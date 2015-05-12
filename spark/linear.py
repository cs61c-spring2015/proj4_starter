import numpy as np
import cPickle as pickle
from classifier import Classifier
from util.layers import *
from util.dump import *

""" STEP1: Build Linear Classifier """

class LinearClassifier(Classifier):
  def __init__(self, D, H, W, K, iternum):
    Classifier.__init__(self, D, H, W, K, iternum)
    """ Parameters """
    # weight matrix: [M * K]
    self.A = 0.01 * np.random.randn(self.M, K)
    # bias: [1 * K]
    self.b = np.zeros((1,K))

    """ Hyperparams """
    # learning rate
    self.rho = 1e-5
    # momentum
    self.mu = 0.9
    # reg strength
    self.lam = 1e1
    # velocity for A: [M * K]
    self.v = np.zeros((self.M, K))
    return

  def load(self, path):
    data = pickle.load(open(path + "layer1"))
    assert(self.A.shape == data['w'].shape)
    assert(self.b.shape == data['b'].shape)
    self.A = data['w']
    self.b = data['b']
    return

  def param(self):
    return [("A", self.A), ("b", self.b)]
 
  def forward(self, data):
    """
    INPUT:
      - data: RDD[(key, (images, labels)) pairs]
    OUTPUT:
      - RDD[(key, (images, list of layers, labels)) pairs]
    """
    """ 
    Layer 1: linear 
    Todo: Implement the forward pass of Layer1
    """
    return data.map(lambda (k, (x, y)): (k, (x, [np.zeros((x.shape[0], 2))], y))) # Replace it with your code

  def backward(self, data, count):
    """
    INPUT:
      - data: RDD[(images, list of layers, labels) pairs]
    OUTPUT:
      - loss
    """

    """ 
    TODO: Implement softmax loss layer
    (images, scores, labels pairs -> (images, (loss, gradient))
    """
 
    """ 
    Todo: Implement backpropagation for Layer 1 
    """

    """
    Todo: Compute the loss and the gradients A & B
    Hint: You need to reduce the RDD from 'backpropagation for Layer 1'
          Also check the output of the loss and the backward function
    """
    L = 0.0
    dLdA = np.zeros(self.A.shape)
    dLdb = np.zeros(self.b.shape)

    """ gradient scaling """
    L /= float(count)
    dLdA /= float(count)
    dLdb /= float(count)

    """ regularization: loss = 1/2 * lam * sum_nk(A_nk * A_nk) """
    L += 0.5 * self.lam * np.sum(self.A * self.A) 

    """ regularization gradient """
    dLdA = dLdA.reshape(self.A.shape)
    dLdA += self.lam * self.A

    """ tune the parameter """
    self.v = self.mu * self.v - self.rho * dLdA
    self.A += self.v
    self.b += -self.rho * dLdb
   
    return L
