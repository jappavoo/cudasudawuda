#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <inttypes.h>
#include <time.h>
#include <stdbool.h>

// uncomment to see debug prints from first thread in everny block
// #define BDEBUG

#ifdef BDEBUG
#define BPRINT(fmt, ...) if (threadIdx.x == 0) {			\
    printf("(%d,%d,%d).(%d,%d,%d): %s: " fmt,				\
	   blockIdx.x, blockIdx.y, blockIdx.z,				\
	   threadIdx.x, threadIdx.y, threadIdx.z, __func__, __VA_ARGS__); \
  }
#else
#define BPRINT(...)
#endif

#define NYI { fprintf(stderr, "%s: %d: NYI\n", __func__, __LINE__); assert(0); }
#define CLOCK_SOURCE CLOCK_MONOTONIC
#define NSEC_IN_SECOND (1000000000)
#define TERA_OPS (1000000000000)
#define GIGABYTE (1024ULL * 1024ULL * 1024ULL)
#define KILOBYTE (1024ULL)
typedef struct timespec ts_t;

static inline int
ts_now(ts_t *now)
{
  if (clock_gettime(CLOCK_SOURCE, now) == -1) {
    perror("clock_gettime");
    NYI;
    return 0;
  }
  return 1;
}

static inline uint64_t
ts_diff(ts_t start, ts_t end)
{
  uint64_t diff=((end.tv_sec - start.tv_sec)*NSEC_IN_SECOND) + (end.tv_nsec - start.tv_nsec);
  return diff;
} 

void getGPUProps(int dev, cudaDeviceProp *deviceProp) {
  cudaError_t rc;
  rc = cudaGetDeviceProperties(deviceProp, dev);
  assert(rc == cudaSuccess);
}

void
dumpGPUProps(int dev, cudaDeviceProp *devProp)
{
  printf("  Device %d: %s\n", dev, devProp->name);
  printf("    Compute Capability: %d.%d\n", devProp->major, devProp->minor);
  printf("    Total Global Memory: %.2f GB\n",
	 (double)devProp->totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
  printf("    Clock Rate: %.2f MHz\n", (double)devProp->clockRate / 1000.0);
  printf("    #SM: %d\n", devProp->multiProcessorCount);
  printf("    Max Threads per SM: %d\n", devProp->maxThreadsPerMultiProcessor);
  printf("    Max Threads per block: %d\n", devProp->maxThreadsPerBlock);
  printf("    #Threads per Warp: %d\n", devProp->warpSize);
}


bool GPUcnt()
{
  cudaError_t rc;
  int cnt;
  
  rc = cudaGetDeviceCount(&cnt);
  if (rc == cudaSuccess && cnt > 0) return cnt;
  return 0;
}

void
h_sum(float *result, float *vec, uint64_t n)
{
  float sum;
  uint64_t i;

  for (i=0; i<n; i++) {
    sum += vec[i];
  }
  *result = sum;
}

void initData(float *data, uint64_t n) {
  uint64_t  i;
  for (i=0; i<n; i++) {
    data[i]=(i%2) ? 1.0 : -1.0; // odd will be positive 1 and even negative 
  }
  data[n-1]=43.0;   // one value that cause final sum to 0 
}


__global__ void reduce1(float *d_ivec, float *d_ovec) {
  extern __shared__ float sdata[];

  unsigned int tid = threadIdx.x;
  unsigned int i   = blockIdx.x * blockDim.x + threadIdx.x;

  BPRINT("%d.%d xdim=%d sdata:%p d_ivec=%p d_ovec=%p\n", blockIdx.x, tid,
	 blockDim.x, sdata, d_ivec, d_ovec);

  // all threads used to copy a data item from global memory to shared memory
  sdata[tid]       = d_ivec[i];
  __syncthreads();

  BPRINT("%d.%d sdata coppied\n",blockIdx.x, tid);
  // all data for this block is in shared memory now
  // we will now do log2 passes with strides doubling each time
  for (unsigned int s=1; s<blockDim.x; s *= 2) {
    // for this stride a stride workers are threads with tids divisible by
    // (2 * stride). Each stride work thread sums its element and  element
    // that is s elements away from it 
    if ( (tid % (2*s)) == 0 ) {
      BPRINT("s:%d sdata[%d]=sdata[%d]\n", s, tid, tid+s);
      sdata[tid] += sdata[tid+s];
    }
    __syncthreads();    // barrier here to make sure all sums are done
  }

  BPRINT("%d.%d sdata computed\n", blockIdx.x, tid);

  if (tid==0) d_ovec[blockIdx.x] = sdata[0];
}

int main(int argc, char **argv) {
  uint64_t  bytes = 4 * GIGABYTE;
  int       n,  numFOps, numGPUS, cpuflg=0;
  ts_t      start, end;
  float     result, *h_vec;

  if (argc>1) {
    if (strncmp(argv[1], "-c", 2)==0) {
      cpuflg = 1;
    } else {
      bytes=atoi(argv[1]) * GIGABYTE;
    }
  }

  n       = bytes/(sizeof(float));
  fprintf(stderr, "reduce: bytes=%lu n=%d\n", bytes, n);
  
  numFOps = n;                     // one floating point operation per element
  assert(n*sizeof(float)==bytes);
  h_vec    = (float *)malloc(bytes);
  assert(h_vec);

  initData(h_vec,n);

  if (cpuflg) {
    assert(ts_now(&start));    
    h_sum(&result, h_vec, n);
    assert(ts_now(&end));
    uint64_t ns=ts_diff(start,end);
    double   sec=(double)ns/(double)NSEC_IN_SECOND;
    double   flops=(double)numFOps/sec;
    printf("host: %f,%lu,%lu,%lf,%lf,%lf\n", result, numFOps, ns, sec,
	   flops,flops/TERA_OPS);
  }
  
  numGPUS = GPUcnt();
  
  if (numGPUS>0) {
    assert(numGPUS==1);
    cudaDeviceProp devprops;
    cudaError_t rc;
    void *d_mem = NULL;
    float *d_ivec, *d_ovec, dresult;
    const int blksize=256;
    int gridsize;
    
    getGPUProps(0, &devprops);
    dumpGPUProps(0, &devprops);

    // not sure all corner cases are handled when n is not a multiple
    // of blocksize for the moment check and assert that it is 
    assert( ((n+blksize-1)/blksize)*blksize == n );
    
    // all reductions will consume at most the same space (-1)
    rc = cudaMalloc((void **)&d_mem, 2*bytes); 
    assert(rc == cudaSuccess);
    d_ivec = (float *)d_mem;   
    d_ovec = d_ivec;         // initial "last" output to be the first input
    
    fprintf(stderr, "h_vec=%p d_mem=%p, d_ivec=%p d_ovec=%p"
	    " bytes=%lu 2*bytes=%lu n=%u\n",
	    h_vec, d_mem, d_ivec, d_ovec, bytes, 2*bytes, n);
    rc = cudaMemcpy(d_mem, h_vec, bytes, cudaMemcpyHostToDevice);
    assert(rc == cudaSuccess);

    // Reduce will produce 1 value for each block
    // loop until only 1 value left (output from a reduce on 1 block)
    for (int len=n; len>1; len=len/blksize) {
      d_ivec   = d_ovec;
      d_ovec   = &(d_ivec[len]);  // place output to right of last input element
      gridsize = (len + blksize - 1) / blksize; // ceil(len/blksize)
      fprintf(stderr, "%d: reduce1<<<%d,%d,%d>>>(d_ivec=%p, d_ovec=%p)\n",
	      len, gridsize, blksize, blksize*sizeof(float), d_ivec, d_ovec);
      reduce1<<<gridsize,blksize,blksize*sizeof(float)>>>(d_ivec, d_ovec);
      rc = cudaGetLastError();
      if (rc != cudaSuccess) {
	fprintf(stderr, "CUDA kernel launch failed: %s\n",
		cudaGetErrorString(rc));
	exit(EXIT_FAILURE);
      }
    }
    
    rc = cudaDeviceSynchronize();
    if (rc != cudaSuccess) {
      fprintf(stderr, "cudaDeviceSynchronize failed: %s\n",
	      cudaGetErrorString(rc));
      exit(EXIT_FAILURE);
    }

    assert(rc==cudaSuccess);
    
    rc = cudaMemcpy(&dresult, d_ovec, sizeof(float), cudaMemcpyDeviceToHost);
    if (rc != cudaSuccess) {
      fprintf(stderr, "CUDAMemcpy failed (%p,%p,%d,cudaMemcpyDeviceToHost) dmem=%p (%lu): %s\n",
	      &dresult, d_ovec, sizeof(float), d_mem, 2*bytes,
	      cudaGetErrorString(rc));
	exit(EXIT_FAILURE);
    }
    assert(rc==cudaSuccess);
    rc = cudaFree(d_mem);

    if (cpuflg && result != dresult) {
      fprintf(stderr, "ERROR: result = %f != *h_dresult = %f\n", result, dresult);
    }
    printf("device: %f\n", dresult);
  }
  
  free(h_vec);
  return 0;
}
