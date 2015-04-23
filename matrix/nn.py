import numpy as np
import cPickle as pickle
from classifier import Classifier
from util.layers import *
from util.dump import dump_big_matrix

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
    return [self.A1, self.b1, self.A3, self.b3]

  def forward(self, X, dump_chunks = -1):
    A1 = self.A1
    b1 = self.b1
    A3 = self.A3
    b3 = self.b3

    """
    Layer 1 : linear
    Layer 2 : ReLU
    Layer 3 : linear
    """
    layer1 = linear_forward(X, A1, b1)
    if dump_chunks > 0:
      dump_big_matrix(layer1, "nn_l1_mat", dump_chunks)
    layer2 = ReLU_forward(layer1)
    if dump_chunks > 0:
      dump_big_matrix(layer2, "nn_l2_mat", dump_chunks)
    layer3 = linear_forward(layer2, A3, b3)
    if dump_chunks > 0:
      dump_big_matrix(layer3, "nn_l3_mat", dump_chunks)
    return [layer1, layer2, layer3]

  def backward(self, X, layers, Y, dump_chunks = -1):
    A1 = self.A1
    b1 = self.b1
    A3 = self.A3
    b3 = self.b3
    layer1, layer2, layer3 = layers

    """ softmax classification """
    L, dLdl3 = softmax_loss(layer3, Y)
    if dump_chunks > 0:
      dump_big_matrix(dLdl3, "nn_dLdl3_mat", dump_chunks)

    """ regularization """
    L += 0.5 * self.lam * (np.sum(A1*A1) + np.sum(A3*A3))

    """ backpropagation for Layer 3 """
    dLdl2, dLdA3, dLdb3 = linear_backward(dLdl3, layer2, A3)
    if dump_chunks > 0:
      dump_big_matrix(dLdl2, "nn_dLdl2_mat", dump_chunks)
      dump_big_matrix(dLdA3, "nn_dLdA3_mat", 1)
      dump_big_matrix(dLdb3, "nn_dLdA3_mat", 1)

    """ backpropagation for Layer 2 """
    dLdl1 = ReLU_backward(dLdl2, layer1)
    if dump_chunks > 0:
      dump_big_matrix(dLdl1, "nn_dLdl1_mat", dump_chunks)

    """ backpropagation for Layer 1 """
    dLdX, dLdA1, dLdb1 = linear_backward(dLdl1, X, A1)
    if dump_chunks > 0:
      dump_big_matrix(dLdX, "nn_dLdX_mat", dump_chunks)
      dump_big_matrix(dLdA1, "nn_dLdA1_mat", 1)
      dump_big_matrix(dLdb1, "nn_dLdA1_mat", 1)

    """ regularization gradient """
    dLdA3 = dLdA3.reshape(A3.shape)
    dLdA1 = dLdA1.reshape(A1.shape)
    dLdA3 += self.lam * A3
    dLdA1 += self.lam * A1

    """ tune the parameter """
    self.v1 = self.mu * self.v1 - self.rho * dLdA1
    self.v3 = self.mu * self.v3 - self.rho * dLdA3
    self.A1 += self.v1
    self.A3 += self.v3
    self.b1 += - self.rho * dLdb1
    self.b3 += - self.rho * dLdb3

    return L
