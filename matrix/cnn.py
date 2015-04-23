import numpy as np
import cPickle as pickle
from time import time
from classifier import Classifier
from util.layers import *
from util.dump import dump_big_matrix

class CNNClassifier(Classifier):
  def __init__(self, D, H, W, K, iternum, verbose = False):
    Classifier.__init__(self, D, H, W, K, iternum)
    self.verbose = verbose

    """ 
    Layer 1 Parameters (Conv 32 x 32 x 16) 
    K = 16, F = 5, S = 1, P = 2
    weight matrix: [K1 * D * F1 * F1]
    bias: [K1 * 1]
    """
    K1, F1, self.S1, self.P1 = 16, 5, 1, 2
    self.A1 = 0.01 * np.random.randn(K1, D, F1, F1)
    self.b1 = np.zeros((K1, 1))
    H1 = (H - F1 + 2*self.P1) / self.S1 + 1
    W1 = (W - F1 + 2*self.P1) / self.S1 + 1

    """ 
    Layer 3 Parameters (Pool 16 x 16 x 16) 
    K = 16, F = 2, S = 2
    """
    K3, self.F3, self.S3 = K1, 2, 2
    H3 = (H1 - self.F3) / self.S3 + 1
    W3 = (W1 - self.F3) / self.S3 + 1
 
    """ 
    Layer 4 Parameters (Conv 16 x 16 x 20) 
    K = 20, F = 5, S = 1, P = 2
    weight matrix: [K4 * K3 * F4 * F4]
    bias: [K4 * 1]
    """
    K4, F4, self.S4, self.P4 = 20, 5, 1, 2
    self.A4 = 0.01 * np.random.randn(K4, K3, F4, F4)
    self.b4 = np.zeros((K4, 1))
    H4 = (H3 - F4 + 2*self.P4) / self.S4 + 1
    W4 = (W3 - F4 + 2*self.P4) / self.S4 + 1

    """ 
    Layer 6 Parameters (Pool 8 x 8 x 20) 
    K = 20, F = 2, S = 2
    """
    K6, self.F6, self.S6 = K4, 2, 2
    H6 = (H4 - self.F6) / self.S6 + 1
    W6 = (W4 - self.F6) / self.S6 + 1

    """ 
    Layer 7 Parameters (Conv 8 x 8 x 20) 
    K = 20, F = 5, S = 1, P = 2
    weight matrix: [K7 * K6 * F7 * F7]
    bias: [K7 * 1]
    """
    K7, F7, self.S7, self.P7 = 20, 5, 1, 2
    self.A7 = 0.01 * np.random.randn(K7, K6, F7, F7)
    self.b7 = np.zeros((K7, 1))
    H7 = (H6 - F7 + 2*self.P7) / self.S7 + 1
    W7 = (W6 - F7 + 2*self.P7) / self.S7 + 1

    """ 
    Layer 9 Parameters (Pool 4 x 4 x 20) 
    K = 20, F = 2, S = 2
    """
    K9, self.F9, self.S9 = K7, 2, 2
    H9 = (H7 - self.F9) / self.S9 + 1
    W9 = (W7 - self.F9) / self.S9 + 1

    """ 
    Layer 10 Parameters (FC 1 x 1 x K)
    weight matrix: [(K6 * H_6 * W_6) * K] 
    bias: [1 * K]
    """
    self.A10 = 0.01 * np.random.randn(K9 * H9 * W9, K)
    self.b10 = np.zeros((1, K))

    """ Hyperparams """
    # learning rate
    self.rho = 1e-2
    # momentum
    self.mu = 0.9
    # reg strength
    self.lam = 0.1
    # velocity for A1: [K1 * D * F1 * F1]
    self.v1 = np.zeros((K1, D, F1, F1))
    # velocity for A4: [K4 * K3 * F4 * F4]
    self.v4 = np.zeros((K4, K3, F4, F4))
    # velocity for A7: [K7 * K6 * F7 * F7]
    self.v7 = np.zeros((K7, K6, F7, F7))
    # velocity for A10: [(K9 * H9 * W9) * K]   
    self.v10 = np.zeros((K9 * H9 * W9, K))
 
    return

  def load(self, path):
    data = pickle.load(open(path + "layer1"))
    assert(self.A1.shape == data['w'].shape)
    assert(self.b1.shape == data['b'].shape)
    self.A1 = data['w']
    self.b1 = data['b'] 
    data = pickle.load(open(path + "layer4"))
    assert(self.A4.shape == data['w'].shape)
    assert(self.b4.shape == data['b'].shape)
    self.A4 = data['w']
    self.b4 = data['b']
    data = pickle.load(open(path + "layer7"))
    assert(self.A7.shape == data['w'].shape)
    assert(self.b7.shape == data['b'].shape)
    self.A7 = data['w']
    self.b7 = data['b']
    data = pickle.load(open(path + "layer10"))
    assert(self.A10.shape == data['w'].shape)
    assert(self.b10.shape == data['b'].shape)
    self.A10 = data['w']
    self.b10 = data['b']
    return 

  def param(self):
    return [
      self.A10, self.b10,
      self.A7, self.b7, 
      self.A4, self.b4, 
      self.A1, self.b1]

  def forward(self, X, dump_chunks = -1):
    A1 = self.A1
    b1 = self.b1
    S1 = self.S1
    P1 = self.P1

    F3 = self.F3
    S3 = self.S3

    A4 = self.A4
    b4 = self.b4
    S4 = self.S4
    P4 = self.P4

    F6 = self.F6
    S6 = self.S6

    A7 = self.A7
    b7 = self.b7
    S7 = self.S7
    P7 = self.P7

    F9 = self.F9
    S9 = self.S9

    A10 = self.A10
    b10 = self.b10

    s = time()
    layer1, X_col1 = conv_forward(X, A1, b1, S1, P1)
    e = time()
    if self.verbose:
      print """ Layer1: Conv (32 x 32 x 16) forward doen: %.2f sec """ % (e - s)
    if dump_chunks > 0:
      dump_big_matrix(layer1, "cnn_l1_mat", dump_chunks)
 
    s = time()
    layer2 = ReLU_forward(layer1)
    e = time()
    if self.verbose:
      print """ Layer2: ReLU (32 x 32 x 16) forward done: %.2f sec """ % (e - s)
    if dump_chunks > 0:
      dump_big_matrix(layer2, "cnn_l2_mat", dump_chunks)

    s = time()
    layer3, X_idx3 = max_pool_forward(layer2, F3, S3)
    e = time()
    if self.verbose:
      print """ Layer3: Pool (16 x 16 x 16) forward done: %.2f sec """ % (e - s)
    if dump_chunks > 0:
      dump_big_matrix(layer3, "cnn_l3_mat", dump_chunks)

    s = time()
    layer4, X_col4 = conv_forward(layer3, A4, b4, S4, P4)
    e = time()
    if self.verbose:
      print """ Layer4: Conv (16 x 16 x 20) forward done: %.2f sec """ % (e - s)
    if dump_chunks > 0:
      dump_big_matrix(layer4, "cnn_l4_mat", dump_chunks)

    s = time()
    layer5  = ReLU_forward(layer4)
    e = time()
    if self.verbose:
      print """ Layer5: ReLU (16 x 16 x 20) forward done: %.2f sec """ % (e - s)
    if dump_chunks > 0:
      dump_big_matrix(layer5, "cnn_l5_mat", dump_chunks)

    s = time()
    layer6, X_idx6 = max_pool_forward(layer5, F6, S6)
    e = time()
    if self.verbose:
      print """ Layer6: Pool (8 x 8 x 20) forward done: %.2f sec """ % (e - s)
    if dump_chunks > 0:
      dump_big_matrix(layer6, "cnn_l6_mat", dump_chunks)

    s = time()
    layer7, X_col7 = conv_forward(layer6, A7, b7, S7, P7)
    e = time()
    if self.verbose:
      print """ Layer7: Conv (8 x 8 x 20) forward: %.2f sec """ % (e - s)
    if dump_chunks > 0:
      dump_big_matrix(layer7, "cnn_l7_mat", dump_chunks)

    s = time()
    layer8 = ReLU_forward(layer7)
    e = time()
    if self.verbose:
      print """ Layer8: ReLU (8 x 8 x 20) forward done: %.2f sec """ % (e - s)
    if dump_chunks > 0:
      dump_big_matrix(layer8, "cnn_l8_mat", dump_chunks)

    s = time()
    layer9, X_idx9 = max_pool_forward(layer8, F9, S9)
    e = time()
    if self.verbose:
      print """ Layer9: Pool (4 x 4 x 20) forward done: %.2f sec """ % (e - s)
    if dump_chunks > 0:
      dump_big_matrix(layer9, "cnn_l9_mat", dump_chunks)

    s = time()
    layer10 = linear_forward(layer9, A10, b10)
    e = time()
    if self.verbose:
      print """ Layer10: FC (1 x 1 x 10) forward done: %.2f sec """ % (e - s)
    if dump_chunks > 0:
      dump_big_matrix(layer10, "cnn_l10_mat", dump_chunks)

    return [
      (layer1, X_col1), layer2, (layer3, X_idx3),
      (layer4, X_col4), layer5, (layer6, X_idx6),
      (layer7, X_col7), layer8, (layer9, X_idx9), layer10]

  def backward(self, X, layers, Y, dump_chunks = -1):
    A1 = self.A1
    b1 = self.b1
    S1 = self.S1
    P1 = self.P1

    F3 = self.F3
    S3 = self.S3

    A4 = self.A4
    b4 = self.b4
    S4 = self.S4
    P4 = self.P4

    F6 = self.F6
    S6 = self.S6

    A7 = self.A7
    b7 = self.b7
    S7 = self.S7
    P7 = self.P7

    F9 = self.F9
    S9 = self.S9

    A10 = self.A10
    b10 = self.b10

    (layer1, X_col1), layer2, (layer3, X_idx3), \
    (layer4, X_col4), layer5, (layer6, X_idx6), \
    (layer7, X_col7), layer8, (layer9, X_idx9), layer10 = layers

    s = time()
    L, dLdl10 = softmax_loss(layer10, Y)
    e = time()
    if self.verbose:
      print """ Softmax loss calc done : %.2f sec """ % (e - s)
    if dump_chunks > 0:
      dump_big_matrix(dLdl10, "cnn_dLdl10_mat", dump_chunks)

    """ regularization """
    L += 0.5 * self.lam * np.sum(A1*A1)
    L += 0.5 * self.lam * np.sum(A4*A4)
    L += 0.5 * self.lam * np.sum(A7*A7)
    L += 0.5 * self.lam * np.sum(A10*A10)

    s = time()
    dLdl9, dLdA10, dLdb10 = linear_backward(dLdl10, layer9 , A10)
    e = time()
    if self.verbose:
      print """ Layer10: FC (1 x 1 x 10) backward done: %.2f sec """ % (e - s)
    if dump_chunks > 0:
      dump_big_matrix(dLdl9, "cnn_dLdl9_mat", dump_chunks)
      dump_big_matrix(dLdA10, "cnn_dLdA10_mat", 1)
      dump_big_matrix(dLdb10, "cnn_dLdA10_mat", 1)

    """ Pool (4 x 4 x 20) Backward """
    s = time()
    dLdl8 = max_pool_backward(dLdl9, layer8, X_idx9, F9, S9)
    e = time()
    if self.verbose:
      print """ Layer9: Pool (4 x 4 x 20) backward done: %.2f sec """ % (e - s)
    if dump_chunks > 0:
      dump_big_matrix(dLdl8, "cnn_dLdl8_mat", dump_chunks)

    s = time()
    dLdl7 = ReLU_backward(dLdl8, layer7)
    e = time()
    if self.verbose:
      print """ Layer8: ReLU (8 x 8 x 20) backward done: %.2f sec """ % (e - s)
    if dump_chunks > 0:
      dump_big_matrix(dLdl7, "cnn_dLdl7_mat", dump_chunks)

    s = time()
    dLdl6, dLdA7, dLdb7 = conv_backward(dLdl7, layer6, X_col7, A7, S7, P7)
    e = time()
    if self.verbose:
      print """ Layer7: Conv (8 x 8 x 20) backward done: %.2f sec """ % (e - s)
    if dump_chunks > 0:
      dump_big_matrix(dLdl6, "cnn_dLdl6_mat", dump_chunks)
      dump_big_matrix(dLdA7, "cnn_dLdA7_mat", 1)
      dump_big_matrix(dLdb7, "cnn_dLdA7_mat", 1)

    s = time()
    dLdl5 = max_pool_backward(dLdl6, layer5, X_idx6, F6, S6)
    e = time()
    if self.verbose:
      print """ Layer6: Pool (8 x 8 x 20) backward done: %.2f sec """ % (e - s)
    if dump_chunks > 0:
      dump_big_matrix(dLdl5, "cnn_dLdl5_mat", dump_chunks)

    s = time()
    dLdl4 = ReLU_backward(dLdl5, layer4)
    e = time()
    if self.verbose:
      print """ Layer5: ReLU (16 x 16 x 20) backward done: %.2f sec """ % (e - s)
    if dump_chunks > 0:
      dump_big_matrix(dLdl4, "cnn_dLdl4_mat", dump_chunks)

    s = time()
    dLdl3, dLdA4, dLdb4 = conv_backward(dLdl4, layer3, X_col4, A4, S4, P4)
    e = time()
    if self.verbose:
      print """ Layer4: Conv (16 x 16 x 20) backward done: %.2f sec """ % (e - s)
    if dump_chunks > 0:
      dump_big_matrix(dLdl3, "cnn_dLdl3_mat", dump_chunks)
      dump_big_matrix(dLdA4, "cnn_dLdA4_mat", 1)
      dump_big_matrix(dLdb4, "cnn_dLdA4_mat", 1)

    s = time()
    dLdl2 = max_pool_backward(dLdl3, layer2, X_idx3, F3, S3)
    e = time()
    if self.verbose:
      print """ Layer3: Pool (16 x 16 x 16) backward done: %.2f sec """ % (e - s)
    if dump_chunks > 0:
      dump_big_matrix(dLdl2, "cnn_dLdl2_mat", dump_chunks)

    s = time()
    dLdl1 = ReLU_backward(dLdl2, layer1)
    e = time()
    if self.verbose:
      print """ Layer2: ReLU (32 x 32 x 16) backward done: %.2f sec """ % (e - s)
    if dump_chunks > 0:
      dump_big_matrix(dLdl1, "cnn_dLdl1_mat", dump_chunks)

    s = time()
    dLdX, dLdA1, dLdb1 = conv_backward(dLdl1, X, X_col1, A1, S1, P1)
    e = time()
    if self.verbose:
      print """ Layer1: Conv (32 x 32 x 16) backward done: %.2f sec """ % (e - s)
    if dump_chunks > 0:
      dump_big_matrix(dLdX, "cnn_dLdX_mat", dump_chunks)
      dump_big_matrix(dLdA1, "cnn_dLdA1_mat", 1)
      dump_big_matrix(dLdb1, "cnn_dLdA1_mat", 1)

    """ regularization gradient """
    dLdA10 = dLdA10.reshape(A10.shape)
    dLdA7 = dLdA7.reshape(A7.shape)
    dLdA4 = dLdA4.reshape(A4.shape)
    dLdA1 = dLdA1.reshape(A1.shape)
    dLdA10 += self.lam * A10
    dLdA7 += self.lam * A7
    dLdA4 += self.lam * A4
    dLdA1 += self.lam * A1

    """ tune the parameter """
    self.v1 = self.mu * self.v1 - self.rho * dLdA1
    self.v4 = self.mu * self.v4 - self.rho * dLdA4
    self.v7 = self.mu * self.v7 - self.rho * dLdA7
    self.v10 = self.mu * self.v10 - self.rho * dLdA10
    self.A1 += self.v1
    self.A4 += self.v4 
    self.A7 += self.v7
    self.A10 += self.v10
    self.b1 += -self.rho * dLdb1
    self.b4 += -self.rho * dLdb4
    self.b7 += -self.rho * dLdb7
    self.b10 += -self.rho * dLdb10

    return L

