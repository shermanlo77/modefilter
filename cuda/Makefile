LIB	:= -L$(CUDA_HOME)/lib64 -lcudart -lcurand

NVCCFLAGS	:= -arch=sm_75 --ptxas-options=-v --use_fast_math

all:	empiricalnullfilter

empiricalnullfilter: empiricalnullfilter.cu Makefile
	nvcc -ptx empiricalnullfilter.cu -o empiricalnullfilter.ptx $(NVCCFLAGS) $(LIB)

clean:
	rm -f empiricalnullfilter
