#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <inttypes.h>
#include <time.h>
#include <stdbool.h>


#define NYI { fprintf(stderr, "%s: %d: NYI\n", __func__, __LINE__); assert(0); }
#define CLOCK_SOURCE CLOCK_MONOTONIC
#define NSEC_IN_SECOND (1000000000)
#define TERA_OPS (1000000000000)

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

void
dumpGPUProps(int dev)
{
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  printf("  Device %d: %s\n", dev, deviceProp.name);
  printf("    Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
  printf("    Total Global Memory: %.2f GB\n", (double)deviceProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
  printf("    Clock Rate: %.2f MHz\n", (double)deviceProp.clockRate / 1000.0);
  printf("    #SM: %d\n", deviceProp.multiProcessorCount);
  printf("    Max Threads per SM: %d\n", deviceProp.maxThreadsPerMultiProcessor);
  printf("    Max Threads per block: %d\n", deviceProp.maxThreadsPerBlock);
  printf("    #Threads per Warp: %d\n", deviceProp.warpSize);
}

#define FOPSPERLOOP 2
__global__ void kernel(int n) {
  volatile float v1 = 0.0f;
  volatile int i;

  // pragma is need to trick optimizer
  #pragma unroll 1
  for (i=0; i<n; i++) {
    // based on my guess of the sass code that was produced for compute capacity 7.0
    // I think there is a benefit to having a integer and floating point instruction dual
    // issue ... this is a complete guess
    asm volatile ("add.f32 %0, %0, 1.0;" : "+f"(v1));
    asm volatile ("add.f32 %0, %0, 1.0;" : "+f"(v1));
  }

  // next line is to try and trick optimizer
  // having a dependency on v we hope to avoid it from removing the body
  // of the above for loop
  if (v1) { asm volatile ("mov.s32 %0, %%clock;" : "=r"(i)); }
}

bool GPUcnt()
{
  cudaError_t rc;
  int cnt;
  
  rc = cudaGetDeviceCount(&cnt);
  if (rc == cudaSuccess && cnt > 0) return cnt;
  return 0;
}

int main(int argc, char **argv)
{
  cudaError_t rc;
  int n, blks, thdsPerBlk, smemPerBlk;
  ts_t start, end;

  if (GPUcnt()!=1) {
    fprintf(stderr, "ERRORL Expecting 1 GPU on the system, found %d\n", GPUcnt());
    return EXIT_FAILURE;
  }
  
  if (argc != 5) {
    fprintf(stderr, "occtest <# interations> <# blocks> <# threads per block> <# bytes shared mem per block>\n");
    dumpGPUProps(0);
    return EXIT_FAILURE;
  }

  n          = atoi(argv[1]);
  blks       = atoi(argv[2]);
  thdsPerBlk = atoi(argv[3]);
  smemPerBlk = atoi(argv[4]);

  assert(ts_now(&start));

  kernel<<<blks, thdsPerBlk, smemPerBlk>>>(n);
  rc = cudaGetLastError();
  if (rc != cudaSuccess) {
    // Handle the error, e.g., print error message and exit
    fprintf(stderr, "CUDA kernel launch failed: %s\n", cudaGetErrorString(rc));
    exit(EXIT_FAILURE);
  }
  cudaDeviceSynchronize();
  assert(ts_now(&end));
  
  rc = cudaGetLastError();
  if (rc != cudaSuccess) {
    // Handle the error, e.g., print error message and exit
    fprintf(stderr, "CUDA kernel launch failed: %s\n", cudaGetErrorString(rc));
    exit(EXIT_FAILURE);
  }

  
  cudaDeviceReset();

  {
    uint64_t numFOps=(uint64_t)blks * (uint64_t)thdsPerBlk * (uint64_t)n * FOPSPERLOOP;
    uint64_t ns=ts_diff(start,end);
    double   sec=(double)ns/(double)NSEC_IN_SECOND;
    double   flops=(double)numFOps/sec;
    printf("%lu,%lu,%lf,%lf,%lf\n", numFOps, ns, sec, flops,flops/TERA_OPS);
  }
  
  return EXIT_SUCCESS;
}
