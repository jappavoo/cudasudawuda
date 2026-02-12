
# S001

- Short intro
- get us started
- very basic -- simple programs
- next two will build on and introduce fancier features
- after that two lectures on optimization 

# S002

remind you of GPU architecture
GPU separate device
your program will be composed of two parts
1. CPU/Host
2. GPU/Device

GPU part broken down into "blocks" of threads that will be assigned to SM
Once assigned will execute on that SM for its entirety

Independent memories
CPU memory 
GPU memory

Your job is to prepare data on the CPU
move to GPU 
operate on data 
move results back to CPU

high-level overview of our task

# S003

- fundamentally based on C/C++ extensions to do the task we discussed by invoking the runtime library
- bindings to other languages, but CUDA is the native and will allow us to learn the most about how things work
- I will require that you program in C for the assignments project. You can use anything you like
- NVIDIA docs are good and the examples are good but at first might seem a little more complex
- As always, lots on the net, but be aware and also given how fast things change, GenAI can have trouble
- learning to write code and experiment is your best bet
- Competition is hot with OpenCL still the primary 
- I am told OpenCL to CUDA is very easy
- Our goal is to understand how things work, and CUDA maps very closely to the HW


# S004

- When we write out code there will always be two parts (in the same file)
- functions that we invoke from the CPU on the GPU are called kernels
- every time we invoke/launch a kernel it will spawn some number of threads
- you control both the number and how the are organized
- We must break up / organized our threads into "Thread Blocks" of no greater than 1024 threads per block
- the over all collection called a grid
- thread block is the fundamental unit by which work is assigned to a single SM (no migration)
- SM hardware will break threads into warps that can then execute in SIMD style (as discussed) on the SM's resources

# S005

1. Both host and device code written in C -- can live in the same file!
2. I think the simplicity of this choice and systems orientation part of CUDA's success
3. provided library / API lets you control GPU execution and manage memory : malloc and free style
4. intrinsic functions for syncronization and
5. error handling
we will look at examples of all of this

There is a detailed CUDA reference manual in addition to the programmers guide



# S006

- slightly faulty but we will get into it.
- Note no global infront of main --> host and our standard C runtime called entry point
- note `.cu`
- this code compiles with nvcc the cuda c/c++ compiler
- and it executes and does exactly what we asked it to
- no extra magic -- which is good
- so what about our GPU kernel code

# S007

- quick overview of hybrid tool chain
- CUDA GPU C code is first translated into something like JAVA Byte code or WEBASM to a device independent representation PTX
- then translated into SASS for a specific device
- all stuffed into a fat binary that the CUDA runtime can extract and load into the GPU as needed
- fat binary can have multiple versions
- details are in the manuals 

# S008

Here is a slightly more detailed view
- nvcc splits source
1. use host compiler to produce host binary code
2. gpu compiler : produce .ptx and optionally ptxas to produce device specific cubin (sasm)
- all stuffed into a single fat binary


# S009

supports multiple GPU versions both PTX and CUBIN 

# S010

NVCC manual has the details

# S011

- we invoke/launch the kernel
- looks very similar to invoking a normal function
- but we need an execution configuration to specify the organization of our threads
- one big difference to a normal function call is that a kernel invocation is ASYNC
    - returns immediately
    - work of starting the threads and running happens in parallel
    - but host execution immediately proceeds
- so this code again will compile and run
- but there is a bug do you see it?

# S012


# S013


# S014

- In general whenever you compile a .cu you want to specify what types of PTX code and CUBIN you want to generate and leave in the fat binary
- PTX has a version that can independently change -- Virtual Architecture
- GPU real device changes with devices
- NVIDIA defines what is called the compute capability and with each generation of devices they bump the number with both major and minor bumps 

# S015

- this is what happens from a thread point of view
- after invocation we can go on and do what ever we like on the CPU
- GPU threads execute and can straggle in any way they like and of course execute in any order
- sync blocks waiting for CUDA runtime to signal completion, all threads exit/terminate, of last launch

# S016

- general structure of CUDA program
- six basic steps
- go over then
- mention look from 5 to 2
- lots of evolution to make repeated execution more efficient and flexible and to support multiple kernel execution 

# S017

- so lets tackle these in order
- CUDA strives hard to not break our C mental model
- eg. malloc becomes cudaMalloc
- but call structure modified a little for better error handling
- big thing is that CUDAmalloc returns a device address!
- still just a number but is only valid on the device
- CUDA free takes an address returned by CUDA malloc and does what you expect
- our course this is C so you must be organized and take responsibility for the mistakes you will make ;-)
  

# S018

- dest, src
- your jobs is to ensure the correct type of addresses
- and to correctly specify direction
- First and second are all that we will focus on
- Third, let's do fancier device memory management. But I don't know the efficiency
- last one is not that interesting

# S019


# S021

- CUDA exposes the hardware to the programmer
- our job is to break down our task into threads and organize them in a way to efficiently use the hardware and its features
- we aren't going to focus on optimizations right now 

# S022

- mental shift
- massive number of threads
- so look for loops and trying launching each iteration is a reasonable first step
- this is a little hard to get you head around at first
- but remember SIMD under the hood
- later we will want to perhaps unroll and be fancier 

# S023

- common and most natural approach is to start from a data perspective
- look for array's, matrices and tensors
- think of threads working on some region
- again it is reasonable even to assign a thread per element

# S024

- lets think about a simple array example
- 512 not really worth a gpu but lets walk it rhough
- 4 blocks of 128
- 512 threads
- the blocks can now get assigned to different SMs

# S026

- discuss resource constraints
