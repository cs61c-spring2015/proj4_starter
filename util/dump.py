import numpy as np
import os, os.path

"""
This function dumps a single matrice
"""
def dump_matrix(x, prefix="mat_", chunknum=1):
  """
  Input:
  - x: numpy array
  - prefix: string -> used for file names
  - chunknum: integer -> to split a matrix from matrix versions
  Ouput:
  - Nothing
  """
  f = 'dump/' + prefix + '.csv'
  np.savetxt(f, x.reshape(x.shape[0], -1), fmt="%.10f", delimiter = ',')
  return

"""
This function dumps a list of matcies
"""
def dump_list(l, prefix="l_"):
  """
  Input:
  - l: list of matrices
  - prefix: string -> used for file names
  Ouput:
  - Nothing
  """
  for i, x in enumerate(l):
    if type(x).__module__ == np.__name__:
      dump_matrix(x, prefix + "_" + str(i))
    elif hasattr(x, '__iter__'):
      dump_list(x, prefix + "_" + str(i))
  return

"""
This function dumps all possible matirices inside RDD
e.g. dump_rdd(RDD[[x0, [y0, z0]], [x1, [y1, z1]]])
     --> dump/rdd_0_0.csv (x0), 
         dump/rdd_0_1_0.csv (y0),
         dump/rdd_0_1_1.csv (z0),
         dump/rdd_1_0.csv (x1),
         dump/rdd_1_1_0.csv (y1),
         dump/rdd_1_1_1.csv (z1), 
You may want to use map before passing RDD to this function
"""
def dump_rdd(rdd, prefix="rdd_"):
  """
  Input:
  - rdd: RDD[matrices] 
  - prefix: string -> used for file names
  Ouput:
  - Nothing
  """
  path = 'dump/'
  if not os.path.exists(path):
    os.mkdir(path)
  l = rdd.collect()
  for i, x in enumerate(l):
    if type(x).__module__ == np.__name__:
      dump_matrix(x, prefix + "_" + str(i))
    elif hasattr(x, '__iter__'):
      dump_list(x, prefix + "_" + str(i))
  return

"""
This function dumps a big matrix from matrix versions
You can divide the matix by 'chunknum'
"""
def dump_big_matrix(X, prefix="mat", chunknum=1):
  """
  Input:
  - X: numpy array
  - prefix: string -> used for file names
  - chunknum: integer -> to split a matrix from matrix versions
  Ouput:
  - Nothing
  """
  path = 'dump/'
  if not os.path.exists(path):
    os.mkdir(path)
  l = np.split(X, chunknum)
  for i, x in enumerate(l):
    f = path + prefix + '_' + str(i) + '.csv'
    np.savetxt(f, x.reshape(x.shape[0], -1), fmt="%.10f", delimiter = ',')
  return
