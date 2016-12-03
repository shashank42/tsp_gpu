#ifndef _TSP_RNG_H_
#define _TSP_RNG_H_

#include <curand.h>
#include <curand_kernel.h>


/* this GPU kernel function is used to initialize the random states
*  Come from:
*    http://cs.umw.edu/~finlayson/class/fall14/cpsc425/notes/23-cuda-random.html
*
*/
__global__ void init(unsigned int seed, curandState_t* states) {

  /* we have to initialize the state */
  /* the seed can be the same for each core, here we pass the time in from the CPU */
  /* the sequence number should be different for each core (unless you want all
     cores to get the same sequence of numbers for some reason - use thread id! */
  /* the offset is how much extra we advance in the sequence for each call, can be 0 */
  curand_init(seed,
              blockIdx.x * blockDim.x + threadIdx.x,
              0,
              &states[blockIdx.x * blockDim.x + threadIdx.x]);
}

#endif // _TSP_RNG_H_
