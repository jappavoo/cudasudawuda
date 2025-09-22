CFLAGS += -gencode arch=compute_70,code=sm_70
CFLAGS += -gencode arch=compute_80,code=sm_80
CFLAGS += -gencode arch=compute_90,code=sm_90
CFLAGS += --ptxas-options=-v
TARGETS = hello gpuinfo tid fade matadd transpose reduce occtest

all: ${TARGETS}

%.sass: %.cubin
	nvdisasm -c $<  > $@

%.uncommented.cu: %.cu
	./cstrip $< > $@

hello: hello.cu
	nvcc $(CFLAGS) $< -o $@

gpuinfo: gpuinfo.cu
	nvcc $(CFLAGS) $< -o $@

tid: tid.cu
	nvcc  $(CFLAGS) $< -o $@

fade: fade.cu
	nvcc  $(CFLAGS) $< -o $@

matadd: matadd.cu
	nvcc $(CFLAGS) $< -o $@

transpose: transpose.cu
	nvcc $(CFLAGS) $< -o $@

occtest: occtest.cu
	nvcc  $(CFLAGS) --keep  $< -o $@

occtest.cubin: occtest.cu
	nvcc --cubin -gencode arch=compute_70,code=sm_70  $< -o $@

reduce1: reduce.cu
	nvcc  $(CFLAGS) -DKNUM=1  $< -o $@

clean:
	-${RM} -rf $(wildcard ${TARGETS} *.o *.cubin *.ptx *.cudafe* *.ii *.fatbin* *.reg* *.module_id *.sass jobs)
