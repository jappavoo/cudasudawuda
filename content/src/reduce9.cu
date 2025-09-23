__global__ void reduce9(float *d_ivec, float *d_ovec, unsigned int n) {
  extern __shared__ float sdata[];
  unsigned int tid = threadIdx.x;
  // each thread is responsible for positions in 64 blocks
  // eg. blk0.thread0: blk0[0] and blk1[0],
  //     blk1.thread0: blk2[0] and blk3[0]
  //     etc.
  unsigned int i   = blockIdx.x *
  	             (blockDim.x*64)   // note * 64
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
               ((i+(31*blockDim.x) < n) ? d_ivec[i+(31*blockDim.x)] : 0.0) +
               ((i+(32*blockDim.x) < n) ? d_ivec[i+(32*blockDim.x)] : 0.0) +
               ((i+(33*blockDim.x) < n) ? d_ivec[i+(33*blockDim.x)] : 0.0) +
               ((i+(34*blockDim.x) < n) ? d_ivec[i+(34*blockDim.x)] : 0.0) +
               ((i+(35*blockDim.x) < n) ? d_ivec[i+(35*blockDim.x)] : 0.0) +
               ((i+(36*blockDim.x) < n) ? d_ivec[i+(36*blockDim.x)] : 0.0) +
               ((i+(37*blockDim.x) < n) ? d_ivec[i+(37*blockDim.x)] : 0.0) +
               ((i+(38*blockDim.x) < n) ? d_ivec[i+(38*blockDim.x)] : 0.0) +
               ((i+(39*blockDim.x) < n) ? d_ivec[i+(39*blockDim.x)] : 0.0) +
               ((i+(40*blockDim.x) < n) ? d_ivec[i+(40*blockDim.x)] : 0.0) +
               ((i+(41*blockDim.x) < n) ? d_ivec[i+(41*blockDim.x)] : 0.0) +
               ((i+(42*blockDim.x) < n) ? d_ivec[i+(42*blockDim.x)] : 0.0) +
               ((i+(43*blockDim.x) < n) ? d_ivec[i+(43*blockDim.x)] : 0.0) +
               ((i+(44*blockDim.x) < n) ? d_ivec[i+(44*blockDim.x)] : 0.0) +
               ((i+(45*blockDim.x) < n) ? d_ivec[i+(45*blockDim.x)] : 0.0) +
               ((i+(46*blockDim.x) < n) ? d_ivec[i+(46*blockDim.x)] : 0.0) +
               ((i+(47*blockDim.x) < n) ? d_ivec[i+(47*blockDim.x)] : 0.0) +
               ((i+(48*blockDim.x) < n) ? d_ivec[i+(48*blockDim.x)] : 0.0) +
               ((i+(49*blockDim.x) < n) ? d_ivec[i+(49*blockDim.x)] : 0.0) +
               ((i+(50*blockDim.x) < n) ? d_ivec[i+(50*blockDim.x)] : 0.0) +
               ((i+(51*blockDim.x) < n) ? d_ivec[i+(51*blockDim.x)] : 0.0) +
               ((i+(52*blockDim.x) < n) ? d_ivec[i+(52*blockDim.x)] : 0.0) +
               ((i+(53*blockDim.x) < n) ? d_ivec[i+(53*blockDim.x)] : 0.0) +
               ((i+(54*blockDim.x) < n) ? d_ivec[i+(54*blockDim.x)] : 0.0) +
               ((i+(55*blockDim.x) < n) ? d_ivec[i+(55*blockDim.x)] : 0.0) +
               ((i+(56*blockDim.x) < n) ? d_ivec[i+(56*blockDim.x)] : 0.0) +
               ((i+(57*blockDim.x) < n) ? d_ivec[i+(57*blockDim.x)] : 0.0) +
               ((i+(58*blockDim.x) < n) ? d_ivec[i+(58*blockDim.x)] : 0.0) +
               ((i+(59*blockDim.x) < n) ? d_ivec[i+(59*blockDim.x)] : 0.0) +
               ((i+(60*blockDim.x) < n) ? d_ivec[i+(60*blockDim.x)] : 0.0) +
               ((i+(61*blockDim.x) < n) ? d_ivec[i+(61*blockDim.x)] : 0.0) +
               ((i+(62*blockDim.x) < n) ? d_ivec[i+(62*blockDim.x)] : 0.0) +
               ((i+(63*blockDim.x) < n) ? d_ivec[i+(63*blockDim.x)] : 0.0);
   
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

void reduce9bytes(uint64_t *readbytes, uint64_t *writtenbytes, int n, int blksize)
{ 
  uint64_t rb = 0, wb = 0;
  for (int len=n; len>1; len=len/blksize) {
    rb += len;                   // reads all len elements 
    if (n/blksize <= 1) wb += 1; // one for one block or partial block
    else  wb += len/(64*blksize); // one for every eight blocks
    len = len / 64;               // each step takes care of 64 blocks worth
  }
  rb *= sizeof(float); wb *= sizeof(float);
  *readbytes = rb; *writtenbytes = wb;
}
