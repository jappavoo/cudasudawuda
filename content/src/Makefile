CFLAGS += -gencode arch=compute_70,code=sm_70
CFLAGS += -gencode arch=compute_80,code=sm_80
CFLAGS += -gencode arch=compute_90,code=sm_90
all: hello gpuinfo tid fade

hello: hello.cu
        nvcc $(CFLAGS) $< -o $@

gpuinfo: gpuinfo.cu
        nvcc $(CFLAGS) $< -o $@

tid: tid.cu
        nvcc  $(CFLAGS) $< -o $@

fade: fade.cu
        nvcc  $(CFLAGS) $< -o $@

clean:
        -${RM} -rf $(wildcard hello gpuinfo)
