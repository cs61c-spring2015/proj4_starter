import numpy as np
import cPickle as pickle
from classifier import Classifier
from util.layers import *
from util.dump import dump_big_matrix

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

  def param(self):
    return [self.A, self.b]

  def forward(self, X, dump_chunks = -1):
    A = self.A
    b = self.b
    """ Layer 1: linear activation """
    layer1 = linear_forward(X, A, b)
    if dump_chunks > 0:
      dump_big_matrix(layer1, "lin_l1_mat", dump_chunks)
    return [layer1]

  def backward(self, X, layers, Y, dump_chunks = -1):
    A = self.A
    b = self.b
    layer1 = layers[-1]

    """ softmax classification """
    L, dLdl1 = softmax_loss(layer1, Y)
    if dump_chunks > 0:
      dump_big_matrix(dLdl1, "lin_dLdl1_mat", dump_chunks)

    """ regularization: loss = 1/2 * lam * sum_nk(A_nk * A_nk) """
    L += 0.5 * self.lam * np.sum(A * A) 

    """ backpropagation for Layer 1 """
    dLdX, dLdA, dLdb = linear_backward(dLdl1, X, A)
    if dump_chunks > 0:
      dump_big_matrix(dLdX, "lin_dLdX_mat", dump_chunks)
      dump_big_matrix(dLdA, "lin_dLdA_mat", 1)
      dump_big_matrix(dLdb, "lin_dLdb_mat", 1)

    """ regularization gradient """
    dLdA = dLdA.reshape(A.shape)
    dLdA += self.lam * A

    """ tune the parameter """
    self.v = self.mu * self.v - self.rho * dLdA
    self.A += self.v
    self.b += -self.rho * dLdb

    return L
