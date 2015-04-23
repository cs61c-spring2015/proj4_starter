import numpy as np
#from im2col import *
from im2col_c import *

def linear_forward(X, A, b):
  """
  Input:
  - X: [N * M] -> N data of size M
  - A: [M * K] -> M weights for K classes
  - b: [1 * K] -> bias
  Output:
  - X * A + b: [N * K]
  """
  return np.dot(X.reshape(X.shape[0], -1), A) + b

def linear_backward(df, X, A):
  """
  Input:
  - df: [N * K] -> gradient on the upstream function 'f'
  - X:  [N * M] -> N data of size M
  - A:  [M * K] -> M weights for K classes
  Output:
  - dX: [N * M] -> gradient on X
  - dA: [M * K] -> gradient on A
  - db: [K * 1] -> gradient on b
  """

  # gradient on X: dL/dX = dL/df * df/dX = dL/df * A.T: [N * M]
  dX = np.dot(df, A.T).reshape(X.shape)
  # gradient on A: dL/dA = dL/df * df/dA = X.T * dL/df: [M * K]
  dA = np.dot(X.T, df)
  # gradient on b: dL/db = dL/df * df/db = [sum_n(dL_n/df_k)]: [K * 1]
  db = np.sum(df, axis=0)
  return dX, dA, db

def ReLU_forward(f):
  """
  Input:
  - f: [N * K] -> scores for K classes of N data
  Output:
  - max(0, f): [N * K]
  """
  return np.maximum(0, f)

def ReLU_backward(df, g):
  """
  Input:
  - df: [N * K] -> gradient on the upstream function 'f'
  - g:  [N * K] -> output of LeRU_forward
  Output:
  - [if (g_nk <= 0) 0 else df_nk]: [N * K]
  """
  g_ = g.reshape(df.shape)
  df[g_ <= 0] = 0
  return df.reshape(g.shape)

def softmax_loss(f, Y):
  """
  * Coumputes the loss and its gradient for softmax classification
  Input: 
  - f: [N * K] -> scores of N images for K classes
  - Y: [N * 1] -> labels of N images (range(0, K))
  Output:
  - L: softmax loss
  - df: gradient of L on f
  """
  N = f.shape[0]

  # probs: [p_nk = exp(f_k_n) / sum(exp(f_j_n))]: [N * K]
  p = np.exp(f - np.max(f, axis=1, keepdims=True))
  p /= np.sum(p, axis=1, keepdims=True)
  
  # loss: sum_n(-log(p_ny)) / N, where p_ny = prob of the image n's label
  L = np.sum(-np.log(p[np.arange(N), Y])) / N

  # dL_n/df_k = p_k - delta_Y_n_k, where delta_Y_n_k = if (Y[n] == k) 1 else 0
  # gradient of L on f: dL/df = [dL_n/df_k / N]: [N * K]
  df = p.copy()
  df[np.arange(N), Y] -= 1
  df /= N

  return L, df

def conv_forward(X, A, b, S, P):
  """
  Input:
  - X: [N * D * H * W] -> N image data of size D * H * W
  - A: [K * D * F * F] -> K filters of size D * F * F
  - b: [K * 1]           -> bias
  - S: stride of convolution (integer)
  - P: size of zero padding (integer)
  Output:
  - f: [N * K * H_ * W_] -> activation maps, where
    - H_ = (H - F + 2P)/S + 1
    - W_ = (W - F + 2P)/S + 1
  - X_col: [(D * F * F) * (H_ * W_ * N)] -> column stretched images
  """
  N, _, H, W = X.shape
  K, D, F, _ = A.shape 
  assert (H - F + 2*P) % S == 0, 'wrong convolution params'
  assert (W - F + 2*P) % S == 0, 'wrong convolution params'
  H_ = (H - F + 2*P) / S + 1
  W_ = (W - F + 2*P) / S + 1

  # compute X_col: [(D * F * F) * (H_ * W_ * N)]
  X_col = im2col_forward(X, F, F, S, P)
  # compute f = A * X_col_ + b: [K * (H_ * W_) * N] -> [N * K * (H_ * W_)]
  f = np.dot(A.reshape(K, -1), X_col) + b
  f = f.reshape(K, H_, W_, N).transpose(3,0,1,2)

  return f, X_col

def conv_backward(df, X, X_col, A, S, P):
  """
  Input:
  - df:    [N * K * H_ * W_]             -> gradient on the upstream function 'f'
  - X:     [N * D * H  * W]              -> image
  - X_col: [(D * F * F) * (H_ * W_ * N)] -> column strecthed images
  - A:     [K * D * F * F]               -> K filters of size D * F1 * F2
  - S:     stride of convolution (integer)
  - P:     size of zero padding (integer)
  Output:
  - dX: [N * D * H * W] -> gradient on X
  - dA: [K * D * F * F] -> gradient on A
  - db: [K * 1] -> gradient on b
  """
  N, K, W_, H_ = df.shape
  _, D, F, _ = A.shape
  # dL/dX_col = dL/df * df/dX_col = A * df_ : [(N * H_ * W_) * (D * F * F)]
  dX_col = np.dot(A.reshape(K, -1).T, df.transpose(1,2,3,0).reshape(K, -1))
  # dL/dX : [N * D * H * W]   
  dX = im2col_backward(dX_col, X.shape, F, F, S, P)
  # dL/dA = dL/df * df/dA = X_col_ * df_ : [K * (D * F * F)]
  dA = np.dot(X_col, df.transpose(2,3,0,1).reshape(-1, K))
  # dL/db = sum(dL/df): [K * 1]
  db = np.sum(df, axis=(0, 2, 3)).reshape(K, 1)
 
  return dX, dA, db

def max_pool_forward(X, F, S):
  """
  Input:
  - X: [N * D * H * W] -> N image data of size D * H * W
  - F: size of pooling
  - S: stride of pooling
  Output:
  - f: [N * D * H_ * W_] -> pooled maps, where
    - H_ = (H - F)/S + 1
    - W_ = (W - F)/S + 1
  - X_idx: [(H_ * W_ * N * D) * 1] -> indices for X_col
  """
  N, D, H, W = X.shape
  assert (H - F) % S == 0, 'wrong pooling params'
  assert (W - F) % S == 0, 'wrong pooling params'
  H_ = (H - F) / S + 1
  W_ = (W - F) / S + 1

  # compute X_col : [(F * F) * (H_ * W_ * N * D)] 
  X_col = im2col_forward(X.reshape(N*D, 1, H, W), F, F, S, 0)
  # compute X_idx : [(H_ * W_ * N * D)] 
  X_idx = np.argmax(X_col, axis=0)
  # compute f: [N * D * H_ * W_]
  f = X_col[X_idx, np.arange(X_col.shape[1])].reshape(H_,W_,N,D).transpose(2,3,0,1)
 
  return f, X_idx

def max_pool_backward(df, X, X_idx, F, S):
  """
  Input:
  - df:    [ N * D * H_ * W_ ] -> gradient on the upstream function 'f'
  - X:     [ N * D * H  * W  ] -> images
  - X_idx: [(N * D * H_ * W_)] -> indice from column strecthed image
  - F: size of pooling
  - S: stride of pooling
  Output:
  - dX: [N * D * H * W] -> gradient on X
  """
  N, D, H_, W_ = df.shape
  _, _, H, W = X.shape

  # dX_col : [(F * F) * (H_ * W_ * N * D)] 
  dX_col = np.zeros((F*F, H_*W_*N*D))
  dX_col[X_idx, np.arange(dX_col.shape[1])] = df.transpose(2,3,0,1).flatten()
  # dX : [N * D * H * W]  
  dX = im2col_backward(dX_col, (N*D, 1, H, W), F, F, S, 0).reshape(X.shape)  

  return dX
