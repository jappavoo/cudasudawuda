__global__ void reduce1(float *d_ivec, float *d_ovec) {
  extern __shared__ float sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i   = blockIdx.x * blockDim.x + threadIdx.x;

  BPRINT("%d.%d xdim=%d sdata:%p d_ivec=%p d_ovec=%p\n", blockIdx.x, tid, blockDim.x, sdata, d_ivec, d_ovec);
  // all threads used to copy a data item from global memory to shared memory
  sdata[tid] = d_ivec[i];
  
  __syncthreads();

  BPRINT("%d.%d sdata coppied\n",blockIdx.x, tid);
  // All data for this block is in shared memory now. We no can do log2 passes
  // with strides doubling each time.  Each pass produces a set of output
  // values calcuated by adding pairs of numbers.  Each pass produces half
  // the number of outputs compared to its inputs. To do this each pass
  // uses one thread for each output value thus at each pass half the threads
  // of the prior pass are "deactivated". In the last pass only one worker 
  // (tid=0) will be active and do the final sum.  All values are keep in
  // the shared memory array by updating the works corresponding element.
  for (unsigned int s=1; s<blockDim.x; s *= 2) {
    // for stride s stride workers are threads with tids divisible by
    // (2 * stride). Each stride worker thread does one addition its element and
    // the element that is s elements away from it 
    if ( (tid % (2*s)) == 0 ) {
      BPRINT("s:%d sdata[%d]=%f sdata[%d]=%f\n", s, tid, sdata[tid], tid+s, sdata[tid+s]);
      sdata[tid] += sdata[tid+s];
    }
    // barrier here to make sure all sums are done
    __syncthreads();    
    // now we can move on to the next stride
  }
  
  BPRINT("%d.%d sdata computed\n", blockIdx.x, tid);
  if (tid==0) d_ovec[blockIdx.x] = sdata[0];
}
