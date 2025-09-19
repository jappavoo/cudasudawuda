#include <stdio.h>

void initData(float *m, int n)
{
  for (int i=0; i<n; i++) {
    m[i]=(float)i;
  }
  fprintf(stderr, "%p %d %f\n", m, n, m[n-1]);
}

__global__ void transpose_naive(float *odata, float *idata,
	   int width, int height) {
  unsigned int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int yIndex = threadIdx.y + blockIdx.y * blockDim.y;

  if (xIndex < width && yIndex < height) {
    unsigned int index_in  = xIndex + width * yIndex;
    unsigned int index_out = yIndex + height * xIndex;
    odata[index_out] = idata[index_in];
  }
}

int main(int argc, char **argv) {
  //set matrix size
  int nx = atoi(argv[1]);
  int ny = atoi(argv[2]);
  int noElems = nx*ny;
  int bytes = noElems * sizeof(float);
  
  // alloc memory host-side
  float *h_A  = (float *) malloc(bytes);
  float *h_dAT = (float *) malloc(bytes);

  // init matrices with random data
  initData(h_A, noElems);

  // alloc memory dev-side
  float *d_A, *d_AT;
  cudaMalloc((void **) &d_A, bytes);
  cudaMalloc((void **) &d_AT, bytes);

  //transfer data to dev
  cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);

  // invoke Kernel
  dim3 block(32, 32); // configure
  dim3 grid((nx + block.x-1)/block.x,
	    (ny + block.y-1)/block.y);
  transpose_naive<<<grid, block>>>(d_A, d_AT, nx, ny);
  cudaDeviceSynchronize();
  //copy data back
  cudaMemcpy(h_dAT, d_AT, bytes, cudaMemcpyDeviceToHost);
  // free GPU resources
  cudaFree(d_A); cudaFree(d_AT);
  cudaDeviceReset();
  // check result
  return 0;
}
