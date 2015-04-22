import numpy as np
cimport numpy as np
cimport cython

np.import_array()

cdef extern from "im2col.h":
  void im2col(double *X_col, 
              double *X_pad, 
              int N, int D, int H, int W, 
              int F1, int F2, int S, int P)

def im2col_wrapper(
  np.ndarray[double, ndim=2, mode="c"] X_col,
  np.ndarray[double, ndim=4, mode="c"] X_pad,
  N, D, H, W, F1, F2, S, P):
  im2col(
    <double*> np.PyArray_DATA(X_col),
    <double*> np.PyArray_DATA(np.require(X_pad, requirements=['C','A'])),
    N, D, H, W, F1, F2, S, P
  )
  return

def im2col_forward(X, F1, F2, S, P):
  """
  Helper function for conv_forward & max_pool_forward
  Input:
  - X:  [N * D * H * W] -> images
  - F1: height of convolution
  - F2: width of convolution
  - S:  stride of convolution
  - P:  size of zero padding
  Output:
  - X_col: [(D * F1 * F2) * (H_ * W_ * N)] -> column stretched matrix
  """
  N, D, H, W = X.shape
  H_ = (H - F1 + 2*P) / S + 1
  W_ = (W - F2 + 2*P) / S + 1

  # zero-pad X: [N * D * (H+2P) * (W+2P)]
  X_pad = np.pad(X, ((0,0),(0,0),(P,P),(P,P)), 'constant')
  X_pad = np.require(X_pad.transpose(1,2,3,0), np.float64, ['C','A'])
  # compute X_col: [(D * F1 * F2) * (H_ * W_ * N)]
  X_col = np.require(np.zeros((D*F1*F2, H_*W_*N)), np.float64, ['C', 'A'])
  im2col_wrapper(X_col, X_pad, N, D, H, W, F1, F2, S, P)

  return X_col

cdef extern from "im2col.h":
  void col2im(double *dX_pad, 
              double *dX_col, 
              int N, int D, int H, int W, 
              int F1, int F2, int S, int P)

def col2im_wrapper(
  np.ndarray[double, ndim=4, mode="c"] dX_pad,
  np.ndarray[double, ndim=2, mode="c"] dX_col,
  N, D, H, W, F1, F2, S, P):
  col2im(
    <double*> np.PyArray_DATA(dX_pad),
    <double*> np.PyArray_DATA(dX_col),
    N, D, H, W, F1, F2, S, P
  )
  return

def im2col_backward(dX_col, dim, F1, F2, S, P):
  """
  Helper function for conv_backward & max_pool_backward
  Input:
  - dX_col: [(D * F1 * F2) * (W_ * H_ * N)] -> gradient on X_col
  - dim:    X's dimension(N, D, H, W)
  - F1:     height of convolution
  - F2:     width of convolution
  - S:      stride of convolution
  - P:      size of zero padding
  Output:
  - dX:     [N * D * H * W] -> gradient on X
  """
  N, D, H, W = dim
  H_ = (H - F1 + 2*P) / S + 1
  W_ = (W - F2 + 2*P) / S + 1

  # initialize dX: [N * D * (H+2P) * (W+2P)]
  dX = np.zeros((D, (H+2*P), (W+2*P), N))
  # compute dX: [N * D * H * W]
  col2im_wrapper(dX, np.require(dX_col, np.float64, ['C','A']), N, D, H, W, F1, F2, S, P)
  dX = dX.transpose(3,0,1,2)

  if P > 0:
    dX = dX[:,:,P:-P,P:-P]

  return dX
