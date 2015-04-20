void im2col(
  double *X_col, // reults        : [(D * F1 * F2) * (H_ * W_ * N)]
  double *X_pad, // padded images : [D * H * W * N]
  int N, int D, int H, int W, // img parmas
  int F1, int F2, int S, int P // conv params
);
 
void col2im(
  double *dX_pad, // reults : [N * D * (H + 2P) * (W + 2P)]
  double *dX_col, // im2col : [N * (D * F1 * F2) * (H_ * W_)]
  int N, int D, int H, int W, // img parmas
  int F1, int F2, int S, int P // conv params
); 
