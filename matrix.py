import numpy as np
from util.util import *
from matrix.linear import LinearClassifier
from matrix.nn import NNClassifier
from matrix.cnn import CNNClassifier
from time import time
import sys

if __name__ == '__main__':
  """ parse args """
  name = 'linear'
  datanum = 5000
  if len(sys.argv) > 1:
    name = str(sys.argv[1])
  if len(sys.argv) > 2:
    datanum = int(sys.argv[2])

  """ load imgs """
  classes, X, Y, X_, Y_ = load_images()

  """ sample imgs """
  if name != 'cnn':
    classes = classes[:3]
    X = X[Y < 3, :,:,:]
    Y = Y[Y < 3]
    X_ = X_[Y_ < 3]
    Y_ = Y_[Y_ < 3]

  N = X.shape[0]
  if datanum < N:
    sample = np.random.choice(N, size=datanum, replace=False)   
    X = X[sample]
    Y = Y[sample]
      
  """ set classifiers """
  D = X.shape[1]
  H = X.shape[2]
  W = X.shape[3]
  classifiers = {
    'linear': LinearClassifier(D, H, W, len(classes), 200),
    'nn'    : NNClassifier(D, H, W, len(classes), 50),
    'cnn'   : CNNClassifier(D, H, W, len(classes), 10, True),
  }
  classifier = classifiers[name]

  """ run clssifier """
  log = open('matrix-' + name + '.log', 'w')
  sys.stdout = Log(sys.stdout, log)
  if name == 'cnn':
    classifier.load("snapshot/cnn/")
  s = time()
  classifier.train(X, Y, classes)
  e1 = time()
  classifier.validate(X_, Y_, classes)
  e2 = time()
  print '[CS61C Project 4] time elapsed: %.2f min' % ((e2 - s) / 60.0)
  print '[CS61C Project 4] training performane: %.2f imgs / sec' % \
    ((datanum * classifier.iternum) / (e1 - s))
  log.close()
