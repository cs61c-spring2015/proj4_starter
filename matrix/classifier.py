import numpy as np
from abc import ABCMeta, abstractmethod
from time import time

class Classifier(object):
  """ 
  This is Abstract Base Classs
  You cannot instantiate it
  """
  __metaclass__ = ABCMeta

  def __init__(self, D, H, W, K, iternum):
    self.D = D          # depth of a image (always 3)
    self.H = H          # height of a image
    self.W = W          # width of a image
    self.M = D * H * W  # num of pixels
    self.K = K          # num of classes
    self.iternum = iternum # iter of training
    return

  def preprocess(self, X):
    """ 
    preprocess the data X
    """
    return X / 255.0 - 0.5

  def train(self, X, Y, classes):
    print '[CS61C Project 4] start classifier training'

    X = self.preprocess(X)

    for i in xrange(self.iternum):
      s = time()
      """ evaluate forward values """
      f = self.forward(X)
      """ 
      1) evaluate loss and gradients
      2) tune the parameters
      """
      L = self.backward(X, f, Y)
      e = time()

      print 'iteration: %d, loss: %f, time: %f sec' % (i, L, e - s)
       
    print '[CS61C Project 4] done training'
    return

  def validate(self, X, Y, classes):
    print '[CS61C Project 4] validate the classifier'

    X_ = self.preprocess(X)
    """ evaluate the scores for test images """
    f = self.forward(X_)[-1]
    """ calculate accuracy """
    p = np.argmax(f, axis = 1)

    print '[CS61C Project 4] accuracy: %.2f' % (np.mean(p == Y))
    return

  @abstractmethod
  def load(self, path):
    """
    Load params from 'path'
    """
    pass

  @abstractmethod
  def param(self):
    """
    Return params for tests
    """
    pass

  @abstractmethod
  def forward(self, X):
    """
    Input:
    - X: a numpy matrix of images
    Output:
    - list of all layer forward values
    should be defined in sub-classses
    """
    pass

  @abstractmethod
  def backward(self, layers, Y):
    """
    Input:
    - layers: list of all layer forward values
    Output:
    - Loss value
    this function
    1) computes gradients on each layer and parameter
    2) tunes the parameters
    should be defined in sub-classes
    """
    pass
