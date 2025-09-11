#include <stdio.h>
#include <assert.h>

#define args x[tid]

__device__ float some_function(float x)
{
  return x*x;
}

__global__ void kernel_routine(float *x)
{
   int tid = threadIdx.x +
             blockDim.x * blockIdx.x;

   x[tid] = some_function( args );
}

void filldata(float *data, int n)
{
  for (int i=0; i<n; i++) {
    data[i]=i;
  }
  return;
}

int main(int argc, char **argv) {
   cudaError_t rc;	
   float *h_x, *d_x; // h=host, d=device
   int nblocks=4, nthreads=128;
   int nsize = nblocks * nthreads;

   h_x = (float *) malloc(nsize * sizeof(float));
   rc = cudaMalloc((void **)&d_x, nsize*sizeof(float));
   assert(rc == cudaSuccess);
   
   filldata(h_x, nsize);
   rc = cudaMemcpy( d_x, h_x, nsize*sizeof(float), cudaMemcpyHostToDevice );
   assert(rc == cudaSuccess);
   
   kernel_routine<<<nblocks,nthreads>>>( d_x );
   
   rc = cudaMemcpy( h_x, d_x, nsize*sizeof(float), cudaMemcpyDeviceToHost );
   assert(rc == cudaSuccess);

   for (int n=0; n<nsize; n++) printf("%d,%f\n",n, h_x[n]);

   cudaFree(d_x); free(h_x);
}
