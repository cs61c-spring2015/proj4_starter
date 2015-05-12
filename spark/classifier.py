import numpy as np
from pyspark import StorageLevel
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

  def preprocess(self, data):
    """ 
    preprocess the data X
    """
    return data.map(lambda (k, (x, y)): (k, (x / 255.0 - 0.5, y)))

  def train(self, data, classes, count, is_ec2 = False):
    data = self.preprocess(data)
    print '[CS61C Project 4] start classifier training'

    for i in xrange(self.iternum):
      s = time()
      """ evaluate scores """
      f = self.forward(data).map(lambda v: v[1]).persist(StorageLevel.MEMORY_AND_DISK_SER) \
        if is_ec2 else \
          self.forward(data).map(lambda v: v[1])
      """ 
      1) evaluate loss and gradients 
      2) tune the parameters
      """
      L = self.backward(f, count)
      e = time()

      print 'iteration: %d, loss: %f, time: %f sec' % (i, L, e - s)
       
    print '[CS61C Project 4] done training'
    return

  def validate(self, data, classes, is_ec2 = False):
    print '[CS61C Project 4] test the classifier'

    data = self.preprocess(data)
    """ evaluate the scores for test images """
    f = self.forward(data).map(lambda (k, (x, l, y)): (l[-1], y))\
      .persist(StorageLevel.MEMORY_AND_DISK_SER) if is_ec2 else \
        self.forward(data).map(lambda (k, (x, l, y)): (l[-1], y))
    acc = f.map(lambda (f, y): np.argmax(f, axis=1) == y).cache().mean()
    print '[CS61C Project 4] accuracy: %.2f' % acc
    return

  @abstractmethod
  def load(self):
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
  def forward(self, data):
    """
    Input:
    - X: a numpy matrix of images
    Output:
    - list of all layer forward values
    should be defined in sub-classses
    """
    pass

  @abstractmethod
  def backward(self, data):
    """
    Input:
    - layers: list of all layer forward values
    Output:
    - Loss value
    this function
    1) computes gradients on each layer and parameter
    2) tunes the parametersshould be defined in sub-classes
    """
    pass
