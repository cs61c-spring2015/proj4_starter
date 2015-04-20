import sys
import numpy as np
import cPickle as pickle

if __name__ == '__main__':
  filename = sys.argv[1]
  dim = None
  A = []
  b = []
  with open(filename, 'r') as f:
    i = 0
    mult = 1
    for line in f:
      if i == 0:
        dim = tuple(map(lambda x: int(x), line.split()))
        for k in dim:
          mult *= k 
      elif i < mult + 1:
        A.append(float(line))
      else:
        b.append(float(line))
      i += 1
    A = np.array(A)
    b = np.array(b)
    print dim, A.shape, b.shape
  #print A[0], A[1], A[2], A[3]
  A = A.reshape(dim[3], dim[0], dim[1], dim[2]).transpose(0,3,2,1)
  b = b.reshape((b.shape[0], 1))
  #A = A.reshape((10,4,4,20)).transpose(3,1,2,0).reshape(dim)
  #b = b.reshape((1, b.shape[0]))
  print A.shape, b.shape
  d = {'w': A, 'b': b}
  pickle.dump(d, open("batch", 'w'))
