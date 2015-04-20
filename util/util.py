import numpy as np
import cPickle as pickle

""" Helper class for logging """
class Log(object):
  def __init__(self, *files):
    self.fs = files
  def write(self, obj):
    for f in self.fs:
      f.write(obj)

""" Helper function to load images """
def load_images():
  dirpath = '/home/ff/cs61c/proj3/cifar-10-batches-py/'
  print '[CS61C Project 4] load images'

  """ read classes """
  classes = None
  with open(dirpath + 'batches.meta', 'rb') as f:
    data = pickle.load(f)
    classes = data['label_names']

  """ read train data """
  X = []         # image: [N * D * H * W]
  Y = []         # lable: [N * K(=size(classes))]
  for i in range(1,6):
    filename = 'data_batch_%d' % i
    with open(dirpath + filename, 'rb') as f:
      data = pickle.load(f)
      x = data['data']
      y = data['labels']
      X.append(x.reshape(10000, 3, 32, 32).astype("float"))
      Y.append(y)
  X = np.concatenate(X)
  Y = np.concatenate(Y)

  """ read test data """
  X_ = None       # image
  Y_ = None       # label
  with open(dirpath + 'test_batch', 'rb') as f:
    data = pickle.load(f)
    x = data['data']
    y = data['labels']
    X_ = x.reshape(10000, 3, 32, 32).astype("float")
    Y_ = np.array(y)

  print '[CS61C Project 4] done loading'

  return classes, X, Y, X_, Y_
