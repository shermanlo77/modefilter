LIB	:= -L$(CUDA_HOME)/lib64 -lcudart -lcurand

NVCCFLAGS	:= -arch=sm_75 --ptxas-options=-v --use_fast_math

.PHONY: all clean

all:	cuda python

cuda:	cuda/empiricalnullfilter.ptx

python:	cuda/empiricalnullfilter.ptx
	cp cuda/empiricalnullfilter.ptx python/modefilter/

cuda/empiricalnullfilter.ptx: cuda/empiricalnullfilter.cu Makefile
	nvcc -ptx $< -o $@ $(NVCCFLAGS) $(LIB)

clean:
	rm -f cuda/empiricalnullfilter.ptx
	rm -f python/modefilter/empiricalnullfilter.ptx
