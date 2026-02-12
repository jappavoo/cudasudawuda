# S001 

- Last lec
- most core features we have covered
- Nvidia has been adding higher level features but roughly seem to be largely built out the same core features

# S002

- HW: we know that bastic HW org
   - differs across Compute Capability version of your device — lookup
- SW: kernels, warps, blocks and grids
- GPU memories:  Global, constants,  Shared memory, registers, Local
- kernel invocation, device syncronization
- partitioning your work into thread blocks and implications on occupancy
- prior lectures we started talking about syncronizing threads in a block…. 
   - This lecture more on sync
- remember block execution order is arbitary 
   - and block more or less “sticks to a SM” 

# S003

- nice property is that it ensure both
  1. all threads are at the same point. — the obvious and easy to understand
  2. all writes are visible — includes a memory fence!  more on this


# S004

- Some other fancy syncs that include computation
- but not discussed much  — seems people don’t really use them except the last one 
- count, and, or: Returns the value to all threads


# S005
- Before we go on it is worth noting that there are now some advanced features that extends what we have learnt and perhaps even breaks it
- CGs:  care group threads both with and across blocks, allowing you to barrier in finer granularity
as a matter of fact, even bigger…. 
  - SW abstraction
  - The claim is that hardware accelerates these operations when CC supports them
  - based on HPC I assume some form of combining tree to get logarithm messages exchanged
- Graphs: Allow one to combine CUDA commands from one or more streams into a more complex unit that can be submitted once and rerun without host interactions.

# S006

- while on the topic of synchronization 
- lets look at the common mechanisms and ideas around mutual exclusion
- eg. lock 
  - but remember we want to avoid this as much as possible
- Have included some examples from the older documentation when I could not find a good equivalent

# S007

- Quick basic reminder of why we need mutual exclusion
- Shared counter that we want to increment
- 4 threads of a warp 
- assume value 42 
   -  Add one and then 43
   - All write to count 
- This is not likely what we want.
- We want a critical section around ++
   - one at a time
   
# S008

- Nvidia gpu provide CAS which is very common atomic primitive included on most CPUs
- go over the arguments 
- processor will serialize execution such that parallel execution happens one after the other
- Walk through the  logic
- if you are not familiar this is very standard look it up in an OS textbook

# S009

- Example from NVIDIA is slightly fancier than a typical example, as it adds exponential back off
- walk though
- Note busy wait, and that without the nanosleep, we would be hammering the memory system.
- But the point is that only one thread will hold the mutex; contenders will spin and sleep
- Note might reduce load on memory, but longer latency for waiters to detect release.
- Keep critical sections very small!!!
- We want to avoid trying to use it — use sparingly 
- But there is also a more fundamental problem with this example

# S010

- These systems have what is called a relaxed memory model 
- Just because we released the lock does not mean our memory writes (side effects) have made it
- So other threads might not see what we did!!
- will talk about fences in a couple of sides
- regardless this kind of synchronization is painful/expensice within a warp 
   - avoid having thread’s within a warp use this kind of syncronization
   - eg only have one thread of each warp be a master with respect to what you want to do 
          -- if (threadid % 32) { lock; do work; unlock; }

# S011

- to make our life easier NVIDIA provides a rich set of atomics that avoid the need for a real lock built out of CAS

# S012

- Ok lets be a little more precise — only a little
- Modern world of multi-cores programmers / languages have started to acknowledge the difficults and codify what one can expect from a system’s memory behaviour 
- GPU — relaxed memory — out of order writes
- But the other aspect that NVIDIA has started to codify is the notion of scope
  - that is to say what set of threads — given heterogeneous architecture, are we trying to control
- Can you guess why?
   - waiting for all writes across the entire system might cost a lot more than waiting for writes on my SM
   
# S013

- even more precisely

# S014

- To make sure we are being concrete, incase you have never thought about the implications of out of order execution / weak memory ordering
- Go over Assumptions (top right)
- There are four out comes!!!
   - Note : NOT because two threads are racing to write to the variables
- First note that  Y = 20 comes after X=10 on a single thread — often referred to as program order - and yet we can observe the values that might imply racing with other threads or just a broken world
- note how scope comes in to play
   - threadfence_block — writes visible by all threads on the SM
   - vs  threadfence — writes visible by all threads across all SMs
   - which might be more expensive?
      - so while threadfence would be always safe to use being more precise, if all you need is to coordinate within your block then threadfence_block would be better
- note syncthreads: barrier and fence!

# S015

- if we are trying to ensure memory behaviour we often need to tell the compiler to not optimize ”out” our accesses 
- traditionally in C we use the keyword volatile to mark a variable as potentially changing so that the compiler does not use a register to cache/optimization operations on the variable
   -  but rather always generates memory accesses to operate on the variable.  
   -  eg x=1; y = x; x += 2;
- I had a very hard time understanding to what extent CUDA respects volatile

   
# S016

- Before we move on to talk about exploting asynchrony
- I want to let you know that there are a bunch very interesting primitives for doing synchronized operations within a warp
- can exchange data and compute lane wide global values
- Not going into details — see manual

# S017

- my intuition is that one can creatively use these to squeeze out more performance (but buyer be ware — sleep is also a nice thing)


# S018

-On to our final topic: Classic system optimization
  - Pipelining!
-If you really want to get high scores/performance on the core CUDA assignments this is the key!


# S019

- My main goal is to point out these features so you are aware but not to cover them in depth
- you can read the manuals and look at examples


# S020

- move data between memories on the GPU without using registers and burning threads
- programming interface object-oriented
- create object
    - auto block = cooperateive_groups::this_thread_block();
- to wait
   - block.sync()
- TMA is another new async memory feature that has been added.

# S021

- A big goal has been to evolve the model to avoid the amount of time we  are waiting for I/O 
  - by overlapping I/O between host and GPU (and vice versa)
  - also get compute overlapping 
- Key mechanism is streams

# S022

- remember DMA engine on the GPU do the real memory movement
- and we have lots of compute resources
- the game is to get streams of work going that over lap these things
- commands in a stream are serial
- however across streams they can be concurrent
   - create a stream get a handle and add work to it
   - need 
        - concurrent hardware resources and 
        - to break up your work appropriately 
   - for this to work
   


