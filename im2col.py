import numpy as np

def im2col_indices(D, F1, F2, H_, W_, S):
  """
  Input:
  - D:  size of depth
  - F1: height of convolution
  - F2: width of convolution
  - H_: height of activation maps
  - W_: width of activation maps 
  - S:  stride of convolution
  Output:
  - d:[(D*F1*F2*H_*W_) * 1] -> indices for depth
  - h:[(D*F1*F2*H_*W_) * 1] -> indices for height
  - w:[(D*F1*F2*H_*W_) * 1] -> indices for width
  """
  d = np.array([d_
    for d_ in xrange(D)
    for f1 in xrange(F1)
    for f2 in xrange(F2)
    for h_ in xrange(H_)
    for w_ in xrange(W_)
  ])
  h = np.array([h_*S + f1
    for d_ in xrange(D)
    for f1 in xrange(F1)
    for f2 in xrange(F2)
    for h_ in xrange(H_)
    for w_ in xrange(W_)
  ])
  w = np.array([w_*S + f2
    for d_ in xrange(D)
    for f1 in xrange(F1)
    for f2 in xrange(F2)
    for h_ in xrange(H_)
    for w_ in xrange(W_)
  ])
  return d, h, w

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

  # zero-pad X: [D * (H+2P) * (W+2P) * N]
  X_pad = np.pad(X, ((0,0),(0,0),(P,P),(P,P)), 'constant').transpose(1,2,3,0)
  # get indices for X: [(D*F1*F2*H_*W_) * 1]
  d, h, w = im2col_indices(D, F1, F2, H_, W_, S)
  # compute X_col
  X_col = X_pad[d, h, w, :].reshape(D*F1*F2, -1)

  return X_col

def im2col_backward(dX_col, dim, F1, F2, S, P):
  """
  Helper function for conv_backward & max_pool_backward
  Input:
  - dX_col: [(D * F1 * F2) * (W_ * H_ * N)] -> gradient on X_col
  - dim:    X's dimension(N, W, H, D)
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
  dX = np.zeros((N, D, H + 2*P, W + 2*P))
  # get indices for dX: [(D*F1*F2*H_*W_) * 1]
  d, h, w = im2col_indices(D, F1, F2, H_, W_, S)
  # compute dX: [N * D * H * W]
  np.add.at(dX, (slice(None), d, h, w), dX_col.reshape(-1,N).T)
  if P > 0:
    dX = dX[:,:, P:-P, P:-P] # remove zero-pads

  return dX
