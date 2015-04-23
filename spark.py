import numpy as np
from pyspark import SparkContext
from util.util import *
from spark.linear import LinearClassifier
from spark.nn import NNClassifier
from spark.cnn import CNNClassifier
from time import time
import sys

if __name__ == '__main__':
  """ parse args """
  name = 'linear'
  datanum = 500
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

  D = X.shape[1]
  H = X.shape[2]
  W = X.shape[3]
  N_ = X_.shape[0]

  """ split imgs into smaller chunks """
  X = np.split(X, datanum)
  Y = np.split(Y, datanum)
  X_ = np.split(X_, X_.shape[0])
  Y_ = np.split(Y_, Y_.shape[0])

  """ set classifiers """
  classifiers = {
    'linear': LinearClassifier(D, H, W, len(classes), 200),
    'nn'    : NNClassifier(D, H, W, len(classes), 50),
    'cnn'   : CNNClassifier(D, H, W, len(classes), 10),
  }
  classifier = classifiers[name]

  """ set spark context and RDDs """
  sc = SparkContext()
  trainData = sc.parallelize(zip(xrange(datanum), zip(X, Y)))
  testData = sc.parallelize(zip(xrange(N_), zip(X_, Y_)))

  """ run clssifier """
  log = open('spark-' + name + '.log', 'w')
  sys.stdout = Log(sys.stdout, log)
  if name == 'cnn':
    classifier.load('snapshot/' + name + '/')
  s = time()
  classifier.train(trainData, classes, datanum)
  e1 = time()
  classifier.validate(testData, classes)
  e2 = time()
  print '[CS61C Project 4] training performane: %.2f imgs / sec' % \
    ((datanum * classifier.iternum) / (e1 - s))
  print '[CS61C Project 4] time elapsed: %.2f min' % ((e2 - s) / 60.0)

  trainData.unpersist()
  testData.unpersist()
  sc.stop()
  log.close()
