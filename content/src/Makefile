all: hello gpuinfo
hello: hello.cu
	nvcc $< -o $@

gpuinfo: gpuinfo.cu
	nvcc $< -o $@
