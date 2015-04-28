import numpy as np
from pyspark import SparkContext, SparkConf
from util.util import *
from spark.linear import LinearClassifier
from spark.nn import NNClassifier
from spark.cnn import CNNClassifier
from time import time
import sys
import os

if __name__ == '__main__':
  """ parse args """
  name = 'linear'
  data = 'train'
  datanum = 2000
  if len(sys.argv) > 1:
    name = str(sys.argv[1])
  if len(sys.argv) > 2:
    data = str(sys.argv[2])
  if len(sys.argv) > 3:
    datanum = int(sys.argv[3])

  """ set classifiers """
  classifiers = {
    'linear': LinearClassifier(3, 32, 32, 10, 200),
    'nn'    : NNClassifier(3, 32, 32, 10, 50),
    'cnn'   : CNNClassifier(3, 32, 32, 10, 10),
  }
  classifier = classifiers[name]

  """ set spark context and RDDs """
  master = open("/root/spark-ec2/cluster-url").read().strip()
  slaves = sum(1 for line in open("/root/spark-ec2/slaves"))
  conf = SparkConf()
  conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
  conf.set("spark.python.worker.memory", "50g")
  conf.set("spark.shuffle.consolidateFiles", "true")
  conf.set("spark.default.parallelism", str(slaves * 32))
  sc = SparkContext(master=master, environment={'PYTHONPATH':os.getcwd()}, conf=conf)
  trainData = sc.pickleFile("s3n://61c-cnn/" + data)
  testData = sc.pickleFile("s3n://61c-cnn/test")

  """ run clssifier """
  log = open('ec2-' + name + "-" + data.strip('train-') + '.log', 'w')
  sys.stdout = Log(sys.stdout, log)
  if name == 'cnn':
    classifier.load('snapshot/' + name + '/')
  s = time()
  classifier.train(trainData, [], datanum)
  e1 = time()
  classifier.validate(testData, [])
  e2 = time()
  print '[CS61C Project 4] training performane: %.2f imgs / sec' % \
    ((datanum * classifier.iternum) / (e1 - s))
  print '[CS61C Project 4] time elapsed: %.2f min' % ((e2 - s) / 60.0)

  sc.stop()
  log.close()
