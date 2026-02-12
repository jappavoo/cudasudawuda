# S001

- Continue introduction
- More detail
- More complicated examples 
- Two fundamental topics 
	1. Partitioning and how Maximizing use of the resources 
	2. Memory and Memory Access issues “features”
	
# S002 

- Reminder on some of the vocabulary
1. CPU chip broken down in to SM's
   - SM where threads will execute and can share data efficiently
2. kernel: Term used to refer to a function you write and can be
   "launched"/excuted on the GPU
3. warp: fundamental unit of 32 threads that hardware can schedule
   together to execute the same instruction SIMT/SIMD on an SM
4. blocks: are a grouping threads that will be assigned to execute on
   an SM and schedule as groups of warps
5. grid: the total grouping of all blocks of thread that were launched

- threads, blocks and grids can be expressed as 1,2,or 3D structures
  to make it more natural to map a thread, block and grid to 1,2 or 3D
  data
- Note: for newer hardware with even larger numbers of SM's the is a
  new optional level called a Thread Block Cluster

# S003

- recall quickly general structure of typical CUDA programs
- Go over steps on slide
- Note depending on what we are doing we might need to explicitly
syncronization with work begin done on GPU but implicitly synchronizes
if memcopy follows kernel launch as it comes after the launch in the
"stream" and is a blocking call

# S004

- At a very high-level a programmer's task is to decompose your work into a grid
of threads
  - it to you but typically look to partition loops or data to decompose work into threads
  - we do this by writing a "kernel" function and then launching with parameters that configures a grid of threads that will all execute it
- the typical first thing you do in a kernel is specialize what this thread will uniquely do by computing its id and how that maps to data passed as an argument
   - to enable this CUDA makes available a set of variable you can use -- read from slide
   - a common calucation is to determine this threads global linear id from the variable 
       - using this id lets you determine what work you 
       - go over computation on slide -- computes linear id from provided CUDA 3D variables 
       - Partitioning the linear id space into chucks of 32 lets you know which threads are within the same warp

# S005

- Turns out partitioning work is one of the more difficult things we have to do
- first try and map naturally to the problem structure (eg 2D for matries)
- however we also need to take into account what keep things efficient on the hardware
   - eg. block size should be a multiple of 32 so that warps we execute "full" warps of threads
        - can you see why? 
		    - if a block is towards the end then it will not be a full group of 32
			    - eg if only one thread then 31 wasted threads
		    - not optimial use of hardware
   - maximize # thread per SM to ensure the SM is used efficiently
       - as we will see there are contraints around what resources your threads need
	   - and the resources available on the SM for a given compute capability your device has
   - this includes makeing sure num threads is a multiple of the SM's cores
   - and something we will see over and over again you want to partition your problem and write you code such that global memory (GPU RAM) accesses are coalesced
     - we will discuss more in later slides
- Finally don't forget Amdahl's law -- avoid partitionings and algorithms that require serialization

# S006

Discuss occupancy highlevel -- to achieve good performance our decomposition of the work into threads
needs to not just logically map to the problem but to the hardware such that we maximize the use of 
the SM resources 

# S007

