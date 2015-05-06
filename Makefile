regular     := 2000
small       := 500
large       := 8000
huge        := 20000
full        := 50000
classifiers := linear nn cnn
im2col_src  := $(addprefix util/, im2col_c.pyx im2col.c im2col.h)
im2col      := util/im2col_c.so

$(im2col): $(im2col_src) 
	@cd util && python setup.py build_ext -i

include Makefrag-matrix
include Makefrag-spark
include Makefrag-test
include Makefrag-ec2

clean:
	rm -rf *.pyc matrix/*.pyc spark/*.pyc util/*.pyc test/__init__.pyc *.log dump/
	rm -rf im2col/im2col_c.c util/im2col_c.so util/build

.PHONY: clean
