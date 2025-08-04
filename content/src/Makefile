all: hello gpuinfo
hello: hello.cu
	nvcc $(CFLAGS) $< -o $@

gpuinfo: gpuinfo.cu
	nvcc $(CFLAGS) $< -o $@

clean:
	-${RM} -rf $(wildcard hello gpuinfo)
