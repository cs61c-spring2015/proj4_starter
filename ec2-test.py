import numpy as np
from pyspark import SparkContext, SparkConf, StorageLevel
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
    'linear': LinearClassifier(3, 32, 32, 10, 20),
    'nn'    : NNClassifier(3, 32, 32, 10, 5),
    'cnn'   : CNNClassifier(3, 32, 32, 10, 3),
  }
  classifier = classifiers[name]

  """ set spark context and RDDs """
  master = open("/root/spark-ec2/cluster-url").read().strip()
  slaves = sum(1 for line in open("/root/spark-ec2/slaves"))
  conf = SparkConf()
  conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
  conf.set("spark.eventLog.enabled", "TRUE")
  conf.set("spark.default.parallelism", str(slaves * 2))
  conf.set("spark.akka.frameSize", "50")
  sc = SparkContext(master=master, environment={'PYTHONPATH':os.getcwd()}, conf=conf)
  trainData = sc.pickleFile("s3n://61c-cnn/" + data, slaves * 4)\
                .persist(StorageLevel.MEMORY_AND_DISK_SER)

  """ run clssifier """
  log = open('ec2-' + name + data.strip('train') + '.log', 'w')
  sys.stdout = Log(sys.stdout, log)
  if name == 'cnn':
    classifier.load('snapshot/' + name + '/')
  s = time()
  classifier.train(trainData, [], datanum, is_ec2=True)
  e = time()
  """ skip validation """
  print '[CS61C Project 4] training performane: %.2f imgs / sec' % \
    ((datanum * classifier.iternum) / (e - s))
  print '[CS61C Project 4] time elapsed: %.2f min' % ((e - s) / 60.0)

  trainData.unpersist()
  sc.stop()
  log.close()
