#include <stdio.h>

void initData(float *m, int n)
{
  for (int i=0; i<n; i++) {
    m[i]=(float)i;
  }
  fprintf(stderr, "%p %d %f\n", m, n, m[n-1]);
}

void cmpMats(float *m1, float *m2, int n)
{
  int ecnt=0;
  for (int i=0; i<n; i++) {
    if (m1[i] != m2[i]) {
      ecnt++;
      fprintf(stderr, "%d: m1[%d]=%f != m2[%d]=%f\n",
	      ecnt, i, m1[i], i, m2[i]);
    }
  }
  if (ecnt) {
    fprintf(stderr, "%d mismatches\n", ecnt);
  } else {
    fprintf(stderr, "%f %f\n", m1[n-1], m2[n-1]);
  }
}

void matrixSumHost(float *A, float *B, float *C,
		   int nx, int ny) {
  float *ia=A, *ib=B, *ic=C;
  for (int iy=0; iy<ny; iy++) {
    for (int ix=0; ix<nx; ix++) {
      ic[ix] = ia[ix] + ib[ix];
    }
    ia+=nx; ib+=nx; ic +=nx;
  }
}
__global__ void matrixSumGPU(float *A, float *B, float *C,
			     int nx, int ny) {
  int ix = threadIdx.x + blockIdx.x*blockDim.x;
  int iy = threadIdx.y + blockIdx.y*blockDim.y;
  int idx = iy*nx + ix;
  if ((ix<nx) && (iy<ny)) {
    C[idx] = A[idx] + B[idx];
  }
}

int main(int argc, char **argv) {
  //set matrix size
  int nx = atoi(argv[1]);
  int ny = atoi(argv[2]);
  int noElems = nx*ny;
  int bytes = noElems * sizeof(float);
  // alloc memory host-side
  float *h_A = (float *) malloc(bytes);
  float *h_B = (float *) malloc(bytes);
  float *h_hC = (float *) malloc(bytes); // host result
  float *h_dC = (float *) malloc(bytes); // gpu result
  // init matrices with random data
  initData(h_A, noElems); initData(h_B, noElems);
  // alloc memory dev-side
  float *d_A, *d_B, *d_C;
  cudaMalloc((void **) &d_A, bytes);
  cudaMalloc((void **) &d_B, bytes);
  cudaMalloc((void **) &d_C, bytes);

  //transfer data to dev
  cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
  // invoke Kernel
  dim3 block(32, 32); // configure
  dim3 grid((nx + block.x-1)/block.x,
	    (ny + block.y-1)/block.y);
  matrixSumGPU<<<grid, block>>>(d_A, d_B, d_C, nx, ny);
  cudaDeviceSynchronize();
  //copy data back
  cudaMemcpy(h_dC, d_C, bytes, cudaMemcpyDeviceToHost);
  // free GPU resources
  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
  cudaDeviceReset();
  // check result
  matrixSumHost(h_A, h_B, h_hC, nx, ny);
  cmpMats(h_hC, h_dC, noElems);
  return 0;
}
