__global__ void reduce6(float *d_ivec, float *d_ovec, unsigned int n) {
  extern __shared__ float sdata[];
  unsigned int tid = threadIdx.x;
  // each thread is responsible for positions in eight blocks
  // eg. blk0.thread0: blk0[0] and blk1[0],
  //     blk1.thread0: blk2[0] and blk3[0]
  //     etc.
  unsigned int i   = blockIdx.x *
  	             (blockDim.x*8)   // note * 8 
		     + threadIdx.x;
  
  // don't just copy but accumulate values from global mem into shared memory
  // as long as there are blocks after ours
  sdata[tid] = ((i<n) ? d_ivec[i] : 0.0) +
               ((i+blockDim.x < n) ? d_ivec[i+blockDim.x] : 0.0) +
	       ((i+(2*blockDim.x) < n) ? d_ivec[i+(2*blockDim.x)] : 0.0) +
       	       ((i+(3*blockDim.x) < n) ? d_ivec[i+(3*blockDim.x)] : 0.0) +			
               ((i+(4*blockDim.x) < n) ? d_ivec[i+(4*blockDim.x)] : 0.0) +
	       ((i+(5*blockDim.x) < n) ? d_ivec[i+(5*blockDim.x)] : 0.0) +
       	       ((i+(6*blockDim.x) < n) ? d_ivec[i+(6*blockDim.x)] : 0.0) +
               ((i+(7*blockDim.x) < n) ? d_ivec[i+(7*blockDim.x)] : 0.0);
    
  
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

void reduce6bytes(uint64_t *readbytes, uint64_t *writtenbytes, int n, int blksize)
{ 
  uint64_t rb = 0, wb = 0;
  for (int len=n; len>1; len=len/blksize) {
    rb += len;                   // reads all len elements 
    if (n/blksize <= 1) wb += 1; // one for one block or partial block
    else  wb += len/(8*blksize); // one for every eight blocks
    len = len / 8;               // each step takes care of 8 blocks worth
  }
  rb *= sizeof(float); wb *= sizeof(float);
  *readbytes = rb; *writtenbytes = wb;
}
