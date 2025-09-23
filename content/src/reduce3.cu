__global__ void reduce3(float *d_ivec, float *d_ovec, unsigned int n) {
  extern __shared__ float sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i   = blockIdx.x * blockDim.x + threadIdx.x;

  BPRINT("%d.%d xdim=%d sdata:%p d_ivec=%p d_ovec=%p\n", blockIdx.x, tid, blockDim.x, sdata, d_ivec, d_ovec);
  // all threads used to copy a data item from global memory to shared memory
  sdata[tid] = (i<n) ? d_ivec[i] : 0;
  __syncthreads();

  BPRINT("%d.%d sdata coppied\n",blockIdx.x, tid);

  for (unsigned int s=blockDim.x/2; s>0; s >>= 1) {
    if (tid < s) {  
      sdata[tid] += sdata[tid+s];
    }
    __syncthreads();    
  }

  BPRINT("%d.%d sdata computed: sdata[0]=%f &d_ovec[%d]=%p\n", blockIdx.x, tid, sdata[0], blockIdx.x,  &(d_ovec[blockIdx.x]));
  if (tid==0) d_ovec[blockIdx.x] = sdata[0];
}

void reduce3bytes(uint64_t *readbytes, uint64_t *writtenbytes, int n, int blksize)
{ 
  uint64_t rb = 0, wb = 0;
  for (int len=n; len>1; len=len/blksize) {
    rb += len;
    wb += (len+blksize-1)/blksize; // one value per block or partial block	
  }
  rb *= sizeof(float); wb *= sizeof(float);
  *readbytes = rb; *writtenbytes = wb;
}
