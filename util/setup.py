from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy as np

setup(
  cmdclass={'build_ext': build_ext},
  ext_modules=[
    Extension("im2col_c", 
      sources=["im2col_c.pyx", "im2col.c"], 
      include_dirs=[np.get_include()],
      extra_compile_args=['-std=c99'],
    )
  ]
)
