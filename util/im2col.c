#include "im2col.h"

void im2col(
  double *X_col, // reults        : [(D * F1 * F2) * (H_ * W_ * N)]
  double *X_pad, // padded images : [D * H * W * N]
  int N, int D, int H, int W, // img parmas
  int F1, int F2, int S, int P // conv params
)
{
  int H_ = (H - F1 + 2*P) / S + 1;
  int W_ = (W - F2 + 2*P) / S + 1;

  for (int d = 0 ; d < D ; d++) {
    double *x_col_rgb = X_col + d * (F1 * F2 * H_ * W_ * N);
    double *x_pad_rgb = X_pad + d * ((H+2*P) * (W+2*P) * N);
    for (int f1 = 0 ; f1 < F1 ; f1++) {
      for (int f2 = 0 ; f2 < F2 ; f2++) {
        int col = (f1 * F2 + f2);
        for (int h_ = 0 ; h_ < H_ ; h_++) {
          for (int w_ = 0 ; w_ < W_ ; w_++) {
            int row = (h_ * W_ + w_);
            for (int n = 0 ; n < N ; n++) {
              // X_col[d, f1, f2, h_, w_, n] = X_pad[d, h_*S + f1, w_*S + f2, n] 
              x_col_rgb[n + (row + col * (H_ * W_))*N] = 
              x_pad_rgb[n + ((h_*S + f1) * (W+2*P) + (w_*S + f2))*N];
            }
          }
        }
      }
    }
  }
}

void col2im(
  double *dX_pad, // reults : [N * D * H * W]
  double *dX_col, // im2col : [N * (D * F1 * F2) * (H_ * W_)]
  int N, int D, int H, int W, // img parmas
  int F1, int F2, int S, int P // conv params
)
{
  int H_ = (H - F1 + 2*P) / S + 1;
  int W_ = (W - F2 + 2*P) / S + 1;

  for (int d = 0 ; d < D ; d++) {
    double *dx_pad_rgb = dX_pad + d * ((H+2*P) * (W+2*P) * N);
    double *dx_col_rgb = dX_col + d * (F1 * F2 * H_ * W_ * N);
    for (int f1 = 0 ; f1 < F1 ; f1++) {
      for (int f2 = 0 ; f2 < F2 ; f2++) {
        int col = (f1 * F2 + f2);
        for (int h_ = 0 ; h_ < H_ ; h_++) {
          for (int w_ = 0 ; w_ < W_ ; w_++) {
            int row = (h_ * W_ + w_);
            for (int n = 0 ; n < N ; n++) {
              // dX_pad[d, h_*S + f1, w_*S + f2, n] += dX_col[d, f1, f2, h_, w_, n]
              dx_pad_rgb[n + ((h_*S + f1) * (W+2*P) + (w_*S + f2))*N] += 
              dx_col_rgb[n + (row + col * (H_ * W_))*N];
            }
          }
        }
      }
    }
  }
}
