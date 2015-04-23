regular     := 2000
small       := 500
large       := 8000
classifiers := linear nn cnn

util/im2col_c.so: util/im2col_c.pyx util/im2col.c util/im2col.h
	@cd util && python setup.py build_ext -i

include Makefrag-matrix
include Makefrag-spark
include Makefrag-test

clean:
	rm -rf *.pyc matrix/*.pyc spark/*.pyc util/*.pyc test/__init__.pyc *.log dump/
	rm -rf im2col/im2col_c.c util/im2col_c.so util/build

.PHONY: clean
