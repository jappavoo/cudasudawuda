__global__ void reduce8(float *d_ivec, float *d_ovec, unsigned int n) {
  extern __shared__ float sdata[];
  unsigned int tid = threadIdx.x;
  // each thread is responsible for positions in 32 blocks
  // eg. blk0.thread0: blk0[0] and blk1[0],
  //     blk1.thread0: blk2[0] and blk3[0]
  //     etc.
  unsigned int i   = blockIdx.x *
  	             (blockDim.x*32)   // note * 32
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
               ((i+(7*blockDim.x) < n) ? d_ivec[i+(7*blockDim.x)] : 0.0) +
               ((i+(8*blockDim.x) < n) ? d_ivec[i+(8*blockDim.x)] : 0.0) +
	       ((i+(9*blockDim.x) < n) ? d_ivec[i+(9*blockDim.x)] : 0.0) +
       	       ((i+(10*blockDim.x) < n) ? d_ivec[i+(10*blockDim.x)] : 0.0) +			
               ((i+(11*blockDim.x) < n) ? d_ivec[i+(11*blockDim.x)] : 0.0) +
	       ((i+(12*blockDim.x) < n) ? d_ivec[i+(12*blockDim.x)] : 0.0) +
       	       ((i+(13*blockDim.x) < n) ? d_ivec[i+(13*blockDim.x)] : 0.0) +
               ((i+(14*blockDim.x) < n) ? d_ivec[i+(14*blockDim.x)] : 0.0) +
  	       ((i+(15*blockDim.x) < n) ? d_ivec[i+(15*blockDim.x)] : 0.0) +
	       ((i+(16*blockDim.x) < n) ? d_ivec[i+(16*blockDim.x)] : 0.0) +
       	       ((i+(17*blockDim.x) < n) ? d_ivec[i+(17*blockDim.x)] : 0.0) +			
               ((i+(18*blockDim.x) < n) ? d_ivec[i+(18*blockDim.x)] : 0.0) +
	       ((i+(19*blockDim.x) < n) ? d_ivec[i+(19*blockDim.x)] : 0.0) +
       	       ((i+(20*blockDim.x) < n) ? d_ivec[i+(20*blockDim.x)] : 0.0) +
               ((i+(21*blockDim.x) < n) ? d_ivec[i+(21*blockDim.x)] : 0.0) +
               ((i+(22*blockDim.x) < n) ? d_ivec[i+(22*blockDim.x)] : 0.0) +
	       ((i+(23*blockDim.x) < n) ? d_ivec[i+(23*blockDim.x)] : 0.0) +
       	       ((i+(24*blockDim.x) < n) ? d_ivec[i+(24*blockDim.x)] : 0.0) +			
               ((i+(25*blockDim.x) < n) ? d_ivec[i+(25*blockDim.x)] : 0.0) +
	       ((i+(26*blockDim.x) < n) ? d_ivec[i+(26*blockDim.x)] : 0.0) +
       	       ((i+(27*blockDim.x) < n) ? d_ivec[i+(27*blockDim.x)] : 0.0) +
               ((i+(28*blockDim.x) < n) ? d_ivec[i+(28*blockDim.x)] : 0.0) +
  	       ((i+(29*blockDim.x) < n) ? d_ivec[i+(29*blockDim.x)] : 0.0) +
      	       ((i+(30*blockDim.x) < n) ? d_ivec[i+(30*blockDim.x)] : 0.0) +
               ((i+(31*blockDim.x) < n) ? d_ivec[i+(31*blockDim.x)] : 0.0);	
   
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

void reduce8bytes(uint64_t *readbytes, uint64_t *writtenbytes, int n, int blksize)
{ 
  uint64_t rb = 0, wb = 0;
  for (int len=n; len>1; len=len/blksize) {
    rb += len;                   // reads all len elements 
    if (n/blksize <= 1) wb += 1; // one for one block or partial block
    else  wb += len/(32*blksize); // one for every eight blocks
    len = len / 32;               // each step takes care of 32 blocks worth
  }
  rb *= sizeof(float); wb *= sizeof(float);
  *readbytes = rb; *writtenbytes = wb;
}
