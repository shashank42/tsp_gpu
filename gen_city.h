#ifndef _TSP_GENCITY_H_
#define _TSP_GENCITY_H_

#include <curand.h>
#include <curand_kernel.h>

/* this GPU kernel function is used to initialize the random states */
__global__ void init(unsigned int seed, curandState_t* states) {

  /* we have to initialize the state */
  curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
              blockIdx.x, /* the sequence number should be different for each core (unless you want all
                             cores to get the same sequence of numbers for some reason - use thread id! */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &states[blockIdx.x]);
}



/* Generate random cities in the kernel
Input:
- city_one: [unsigned integer(GRID_SIZE)]
  - Cities to swap for the first swap choice
- city_two: [unsigned integer(GRID_SIZE)] 
  - Cities to swap for the second swap choice
- dist: [float(N * N)]
  - The distance matrix of each city
- salesman_route: [unsigned integer(N)] 
  - The route the salesman will travel
- T: [unsigned integer(1)] 
  - The current temperature
- r: [unsigned integer(GRID_SIZE)]
  - The random number to compare against for S.A.
*/
__global__ void genCity(curandState_t* states, unsigned int* city_one, unsigned int* city_two,
                        unsigned int * N, float * T_start, float *T,
                        unsigned int *flag, volatile unsigned int *global_flag, float *r) {
  /* curand works like rand - except that it takes a state as a parameter */
  int xid = blockIdx.x * blockDim.x + threadIdx.x;
  float myrandf = curand_uniform(states+xid);
  myrandf *= ((N[0]-2) - 0+0.999999);
  myrandf += 0;
  unsigned int myrand = (unsigned int)truncf(myrandf);
  city_one[xid] = myrand;
  float swap_distance = exp(-sqrt(log(T_start[0] / T[0])));
  unsigned int j = (unsigned int)floor(1 + city_one[xid] * swap_distance); 
  city_two[xid] = (city_one[xid] + j) % (N[0]-1);
  
  if (city_two[xid] == 0)
    city_two[xid] += 1;
  flag[xid] = 0;
  global_flag[0] = 0;
  r[xid] = (float)curand_uniform(states+xid);
}
#endif // _TSP_GENCITY_H_

