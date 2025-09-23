#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <inttypes.h>
#include <time.h>
#include <stdbool.h>

// uncomment to see debug prints from first thread in everny block
//#define BDEBUG

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

static inline int ts_now(ts_t *now) {
  if (clock_gettime(CLOCK_SOURCE, now) == -1) {
    perror("clock_gettime");
    NYI;
    return 0;
  }
  return 1;
}

static inline uint64_t ts_diff(ts_t start, ts_t end) {
  uint64_t diff=((end.tv_sec - start.tv_sec)*NSEC_IN_SECOND) +
    (end.tv_nsec - start.tv_nsec);
  return diff;
} 

void getGPUProps(int dev, cudaDeviceProp *deviceProp) {
  cudaError_t rc;
  rc = cudaGetDeviceProperties(deviceProp, dev);
  assert(rc == cudaSuccess);
}

void dumpGPUProps(int dev, cudaDeviceProp *devProp) {
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

bool GPUcnt() {
  cudaError_t rc;
  int cnt;
  
  rc = cudaGetDeviceCount(&cnt);
  if (rc == cudaSuccess && cnt > 0) return cnt;
  return 0;
}

void h_sum(float *result, float *vec, uint64_t n) {
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

#define CONCAT_EXPAND(a, b) a##b
#define CONCAT(a, b) CONCAT_EXPAND(a, b)

#define STRINGIFY_EXPAND(x) #x
#define STRINGIFY(x) STRINGIFY_EXPAND(x)

#define REDUCE_KERNEL(n) CONCAT(reduce, n)
#define REDUCE_KERNEL_STR(n) STRINGIFY(CONCAT(reduce, n))

#include "reduce1.cu"

#include "reduce2.cu"

#include "reduce3.cu"

#include "reduce4.cu"

#include "reduce5.cu"

#include "reduce6.cu"

#include "reduce7.cu"

#include "reduce8.cu"

#include "reduce9.cu"

#define reduceKernel REDUCE_KERNEL(KNUM)
#define reduceBytes  CONCAT(REDUCE_KERNEL(KNUM),bytes) 

int main(int argc, char **argv) {
  uint64_t  bytes = 4 * GIGABYTE;
  int       blksize=256, numtrials=30;		
  int       n, numFOps, numGPUS, cpuflg=0;
  ts_t      hstart, hend, dstart, dend;
  float     result, *h_vec;
  uint64_t  ns, minns, readbytes, writtenbytes, totalbytes;
  double    sec, minsec, bytespersec, minbytespersec, gbpersec, mingbpersec,
            flops, minflops;
  
  
  if (argc>1) {
    if (strncmp(argv[1], "-c", 2)==0) {
      cpuflg = 1;
    } else {
      blksize = atoi(argv[1]);
    }
    if (argc>2) {
      bytes = atoi(argv[2]);
      if (argc>3) {
	numtrials = atoi(argv[3]);
      }
    }
  }
  
  n       = bytes/(sizeof(float));
  fprintf(stderr, REDUCE_KERNEL_STR(KNUM) ": bytes=%lu n=%d blksize=%d\n", bytes,
	  n, blksize);
  
  numFOps = n;                     // one floating point operation per element
  assert(n*sizeof(float)==bytes);
  h_vec    = (float *)malloc(bytes);
  assert(h_vec);

  initData(h_vec,n);

  if (cpuflg) {
    assert(ts_now(&hstart));    
    h_sum(&result, h_vec, n);
    assert(ts_now(&hend));
    ns=ts_diff(hstart,hend);
    sec=(double)ns/(double)NSEC_IN_SECOND;
    flops=(double)numFOps/sec;
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
    int gridsize;

    /*** SETUP ***/
    getGPUProps(0, &devprops);
    dumpGPUProps(0, &devprops);

    printf("Device results\n");
    printf("kernel,trial,dresult,bytes,n,blksize,read,written,totalbytes,"
	   " numFOps, ns, sec, Bytes/s, GB/s, FLOP/s, TFLOP/s\n");

    // not sure all corner cases are handled when n is not a multiple
    // of blocksize for the moment check and assert that it is 
    assert( ((n+blksize-1)/blksize)*blksize == n );
    
    // allocate enough global device memory to hold the input vector
    // and enough memory for all intermediate results produced during
    // all "levels" of the reduction. Which will be at most the
    // same size as the input (-1).  
    rc = cudaMalloc((void **)&d_mem, 2*bytes); 
    assert(rc == cudaSuccess);
    
    // copy vector to start of device memory
    rc = cudaMemcpy(d_mem, h_vec, bytes, cudaMemcpyHostToDevice);
    assert(rc == cudaSuccess);

    for (int trial=0; trial<numtrials; trial++) {
      // init the input and output to both start at the beginning of the data
      d_ivec = (float *)d_mem;
      d_ovec = &(d_ivec[n]);
      /*** Reductions ***/
      // Reduce will produce 1 value for each block
      // loop until only 1 value left (output from a reduce on 1 block)
      assert(ts_now(&dstart));
      for (int len=n; len>1; len=len/blksize) {
	gridsize = (len + blksize - 1) / blksize; // ceil(len/blksize)
#if KNUM == 4
	if (gridsize != 1) {
	  gridsize = gridsize / 2;
	  len      = len / 2;
	}
#elif KNUM == 5
	if (gridsize != 1) {
	  gridsize = gridsize / 4;
	  len      = len / 4;
	}
#elif KNUM == 6
	if (gridsize != 1) {
	  gridsize = gridsize / 8;
	  len      = len / 8;
	}
#elif KNUM == 7
	if (gridsize != 1) {
	  gridsize = gridsize / 16;
	  len      = len / 16;
	}
#elif KNUM == 8
	if (gridsize != 1) {
	  gridsize = gridsize / 32;
	  len      = len / 32;
	}
#elif KNUM == 9
	if (gridsize != 1) {
	  gridsize = gridsize / 64;
	  len      = len / 64;
	}
#endif

	reduceKernel<<<gridsize,blksize,blksize*sizeof(float)>>>(d_ivec,d_ovec,n);
	rc = cudaGetLastError();
	assert(rc == cudaSuccess);
	d_ivec   = d_ovec;
	d_ovec   = &(d_ivec[gridsize]);  // place output to right of last input 
      }
      rc = cudaDeviceSynchronize();
      assert(ts_now(&dend));
      assert(rc == cudaSuccess);
      
      /*** All done get result ***/
      rc = cudaMemcpy(&dresult, d_ivec, sizeof(float), cudaMemcpyDeviceToHost);
      assert(rc==cudaSuccess);
            
      // verify if cpu computation was done
      if (cpuflg && result != dresult) {
	fprintf(stderr, "ERROR: result = %f != *h_dresult = %f\n", result,
		dresult);
      }
      
      // calc performance stats and print
      ns    = ts_diff(dstart,dend);
      sec   = (double)ns/(double)NSEC_IN_SECOND;
      flops = (double)numFOps/sec;
      reduceBytes(&readbytes, &writtenbytes,n,blksize);
      totalbytes = readbytes+writtenbytes;
      bytespersec = (double)totalbytes/(double)sec;
      gbpersec    = bytespersec/(double)1000000000.0;
      printf(REDUCE_KERNEL_STR(KNUM)
	     ",%d,%f,%lu,%lu,%d,%lu,%lu,%lu,%lu,%lu,%lf,%lf,%lf,%lf,%lf\n",
	     trial, dresult, bytes, n, blksize, readbytes, writtenbytes,
	     totalbytes, numFOps, ns, sec, bytespersec,
	     gbpersec, flops, flops/TERA_OPS);
      if (trial==0) {
	minns=ns; minsec=sec; minflops=flops;
      } else {
	if (ns<minns) {
	  minns          = ns;
	  minsec         = sec;
	  minflops       = flops;
	  minbytespersec = bytespersec;
	  mingbpersec    = gbpersec;
	}
      }
      // reset intermediate result memory
      rc = cudaMemset(((char *)d_mem + bytes), 0, bytes);
      assert(rc == cudaSuccess);
      rc = cudaDeviceSynchronize();
      assert(rc == cudaSuccess);
    }
    /* cleanup */
    rc = cudaFree(d_mem);
    assert(rc==cudaSuccess);
    cudaDeviceReset();
    printf(REDUCE_KERNEL_STR(KNUM)
	   ",%d,%f,%lu,%lu,%d,%lu,%lu,%lu,%lu,%lu,%lf,%lf,%lf,%lf,%lf\n",
	   -1, dresult, bytes, n, blksize, readbytes, writtenbytes, totalbytes,
	   numFOps, minns, minsec, minbytespersec, mingbpersec, minflops,
	   minflops/TERA_OPS);
  }
  
  free(h_vec);
  return 0;
}
