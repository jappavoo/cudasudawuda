#include <assert.h>

__global__ void fade( unsigned char *d_in, unsigned char *d_out,
	   	      float f, int xmax, int ymax)  {
  unsigned int idx, v;
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if ((x >= xmax) || ( y >= ymax)) return;

  idx = y * xmax + x; 
  v = d_in[idx] * f;
  if (v>255) v = 255;
  d_out[idx] = v;
}

int main()
{
  cudaError_t rc;	
  int xmax=1024, ymax=1024;
  unsigned char *d_in=NULL, *d_out=NULL;
  float f = 0.5;

  rc = cudaMalloc((void **)d_in, xmax*ymax*sizeof(unsigned char));
  assert(rc == cudaSuccess);
  rc = cudaMalloc((void **)d_out, xmax*ymax*sizeof(unsigned char));
  assert(rc == cudaSuccess);
  
  dim3 nblocks( 7, 6 );
  dim3 nthreads( 16, 16 );
  fade<<<nblocks,nthreads>>>(d_in, d_out, f, xmax, ymax);
}
