CFLAGS += -gencode arch=compute_70,code=sm_70
CFLAGS += -gencode arch=compute_80,code=sm_80
CFLAGS += -gencode arch=compute_90,code=sm_90
CFLAGS += --ptxas-options=-v
TARGETS = hello gpuinfo tid fade matadd transpose reduce1 reduce2 occtest

.PHONY: clean reduce

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

reduce: reduce1 reduce2 reduce3 reduce4 reduce5 reduce6 reduce7 reduce8 reduce9
reduce1: reduce.cu reduce1.cu
	nvcc  $(CFLAGS) -DKNUM=1  $< -o $@

reduce2: reduce.cu reduce2.cu
	nvcc  $(CFLAGS) -DKNUM=2  $< -o $@

reduce3: reduce.cu reduce3.cu
	nvcc  $(CFLAGS) -DKNUM=3  $< -o $@

reduce4: reduce.cu reduce4.cu
	nvcc  $(CFLAGS) -DKNUM=4  $< -o $@

reduce5: reduce.cu reduce5.cu
	nvcc  $(CFLAGS) -DKNUM=5  $< -o $@

reduce6: reduce.cu reduce6.cu
	nvcc  $(CFLAGS) -DKNUM=6  $< -o $@

reduce7: reduce.cu reduce7.cu
	nvcc  $(CFLAGS) -DKNUM=7  $< -o $@

reduce8: reduce.cu reduce8.cu
	nvcc  $(CFLAGS) -DKNUM=8  $< -o $@

reduce9: reduce.cu reduce9.cu
	nvcc  $(CFLAGS) -DKNUM=9  $< -o $@

clean:
	-${RM} -rf $(wildcard ${TARGETS} *.o *.cubin *.ptx *.cudafe* *.ii *.fatbin* *.reg* *.module_id *.sass jobs)
