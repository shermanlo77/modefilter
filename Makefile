NVCC_ARCH	?= sm_86
NVCCFLAGS	:= -arch=$(NVCC_ARCH) --ptxas-options=-v --use_fast_math

.PHONY: all clean

all:	cuda python

build:
	mkdir -p build/cuda

cuda:	build/cuda/empiricalnullfilter.ptx

build/cuda/empiricalnullfilter.ptx: src/cuda/empiricalnullfilter.cu | build
	nvcc -ptx $< -o $@ $(NVCCFLAGS)

python:	build/cuda/empiricalnullfilter.ptx
	cp build/cuda/empiricalnullfilter.ptx src/python/modefilter/

clean:
	rm -rf build
	rm src/python/modefilter/empiricalnullfilter.ptx
