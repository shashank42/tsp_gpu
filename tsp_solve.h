#ifndef _TSP_SOLVE_H_
#define _TSP_SOLVE_H_

#include <curand.h>
#include <curand_kernel.h>

/* TSP With Only Difference Calculation
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
- seed: [unsigned integer(GRID_SIZE)]
- The seed to generate random number
- N [unsigned integer(1)]
- The number of cities.
*/
__global__ static void tsp(unsigned int* city_one,
	                       unsigned int* city_two,
	                       coordinates* __restrict__ location,
	                       unsigned int* __restrict__ salesman_route,
	                       float* __restrict__ T,
	                       int* __restrict__ seed,
	                       volatile unsigned int *global_flag,
	                       unsigned int* __restrict__ N){
    //first, refresh the routine, let thread 0 do it
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tmp;
    if (tid == 0){
        if (global_flag[0] != 0){
            tmp = salesman_route[city_one[global_flag[0]]];
            salesman_route[city_one[global_flag[0]]] = salesman_route[city_two[global_flag[0]]];
            salesman_route[city_two[global_flag[0]]] = tmp;
            global_flag[0] = 0;
        }
    }
    __syncthreads();

    //second, we generate random number, get city_swap_index
    int a_r = 16807, q_r = 127773, z_r = 2836, r_r = seed[tid];
    //note that these parameters' value can't be adjusted,
    // they're optimum, by some mysterious math theory

    r_r = a_r*(r_r%q_r) - z_r*(r_r / q_r);
    if (r_r < 0)
        r_r += 2147483647;               //generate a random number

    int city_one_swap = (int)floor((float)r_r / 2147483647 * N[0]);
    if (city_one_swap == 0)
        city_one_swap += 1;

    r_r = a_r*(r_r%q_r) - z_r*(r_r / q_r);
    if (r_r < 0)
        r_r += 2147483647;                         //generate a new random number

    int city_two_swap = ((int)(city_one_swap +
                  (int)floor(((float)r_r / 2147483647 * 2 - 1) *
                  N[0]*exp(-1 / T[0]))) + N[0]) % N[0];

    if (city_two_swap == 0)
        city_two_swap += 1;
    if (city_two_swap == N[0] - 1)
        city_two_swap -= 1;

    city_one[tid] = city_one_swap;
    city_two[tid] = city_two_swap;

    float delta, p;
    unsigned int trip_city_one = salesman_route[city_one_swap];
    unsigned int trip_city_one_pre = salesman_route[city_one_swap - 1];
    unsigned int trip_city_one_post = salesman_route[city_one_swap + 1];

    unsigned int trip_city_two = salesman_route[city_two_swap];
    unsigned int trip_city_two_pre = salesman_route[city_two_swap - 1];
    unsigned int trip_city_two_post = salesman_route[city_two_swap + 1];
    // The original and post distances
    float original_dist = 0;
    float proposal_dist = 0;

    // We will always have 4 calculations for original distance and the proposed distance, so we just unroll the loop here
    // TODO: It may be nice to make vars for the locations as well so this does not look so gross
    // The first city, unswapped. The one behind it and the one in front of it
    original_dist += (location[trip_city_one_pre].x - location[trip_city_one].x) *
                     (location[trip_city_one_pre].x - location[trip_city_one].x) +
                     (location[trip_city_one_pre].y - location[trip_city_one].y) *
                     (location[trip_city_one_pre].y - location[trip_city_one].y);
    original_dist += (location[trip_city_one_post].x - location[trip_city_one].x) *
                     (location[trip_city_one_post].x - location[trip_city_one].x) +
                     (location[trip_city_one_post].y - location[trip_city_one].y) *
                     (location[trip_city_one_post].y - location[trip_city_one].y);
    // The second city, unswapped. The one behind it and the one in front of it
    original_dist += (location[trip_city_two_pre].x - location[trip_city_two].x) *
                     (location[trip_city_two_pre].x - location[trip_city_two].x) +
                     (location[trip_city_two_pre].y - location[trip_city_two].y) *
                     (location[trip_city_two_pre].y - location[trip_city_two].y);
    original_dist += (location[trip_city_two_post].x - location[trip_city_two].x) *
                     (location[trip_city_two_post].x - location[trip_city_two].x) +
                     (location[trip_city_two_post].y - location[trip_city_two].y) *
                     (location[trip_city_two_post].y - location[trip_city_two].y);
    // The first city, swapped. The one behind it and the one in front of it
    proposal_dist += (location[trip_city_two_pre].x - location[trip_city_one].x) *
                     (location[trip_city_two_pre].x - location[trip_city_one].x) +
                     (location[trip_city_two_pre].y - location[trip_city_one].y) *
                     (location[trip_city_two_pre].y - location[trip_city_one].y);
    proposal_dist += (location[trip_city_two_post].x - location[trip_city_one].x) *
                     (location[trip_city_two_post].x - location[trip_city_one].x) +
                     (location[trip_city_two_post].y - location[trip_city_one].y) *
                     (location[trip_city_two_post].y - location[trip_city_one].y);
    // The second city, swapped. The one behind it and the one in front of it
    proposal_dist += (location[trip_city_one_pre].x - location[trip_city_two].x) *
                     (location[trip_city_one_pre].x - location[trip_city_two].x) +
                     (location[trip_city_one_pre].y - location[trip_city_two].y) *
                     (location[trip_city_one_pre].y - location[trip_city_two].y);
    proposal_dist += (location[trip_city_one_post].x - location[trip_city_two].x) *
                     (location[trip_city_one_post].x - location[trip_city_two].x) +
                     (location[trip_city_one_post].y - location[trip_city_two].y) *
                     (location[trip_city_one_post].y - location[trip_city_two].y);

    //picking the first accepted and picking the last accepted is equivalent, and here I pick the latter one
    //because if I pick the small one, I have to tell whether the flag is 0
    if (proposal_dist < original_dist&&global_flag[0]<tid){
        global_flag[0] = tid;
        __syncthreads();
    } else {
        delta = proposal_dist - original_dist;
        p = exp(-delta / T[0]);
        r_r = a_r*(r_r%q_r) - z_r*(r_r / q_r);
        if (r_r < 0)
            r_r += 2147483647;                         //generate another new random number
        if (p > (float)r_r/2147483547&&global_flag[0]<tid){
            global_flag[0] = tid;
        }
    }
    seed[tid] = r_r;   //refresh the seed at the end of kernel
}


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


/* TSP With Only Difference Calculation
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
- seed: [unsigned integer(GRID_SIZE)]
- The seed to generate random number
- N [unsigned integer(1)]
- The number of cities.
*/
__global__ static void tspLoss(unsigned int* city_one,
	                       unsigned int* city_two,
	                       coordinates* __restrict__ location,
	                       unsigned int* __restrict__ salesman_route,
	                       float* __restrict__ T,
	                       volatile unsigned int *global_flag,
	                       unsigned int* __restrict__ N,
	                       curandState_t* states){
    //first, refresh the routine, let thread 0 do it
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tmp;
    if (tid == 0){
        if (global_flag[0] != 0){
            tmp = salesman_route[city_one[global_flag[0]]];
            salesman_route[city_one[global_flag[0]]] = salesman_route[city_two[global_flag[0]]];
            if (city_one[global_flag[0]] == 0)
                salesman_route[N[0]] = salesman_route[city_two[global_flag[0]]];
            salesman_route[city_two[global_flag[0]]] = tmp;
            global_flag[0] = 0;
        }
    }
    __syncthreads();
    int iter = 0;
    // Wait till global flag is zero and we do 10000 iterations
    while (global_flag[0] == 0 && iter < 10000){
    // Generate the first city
    // From: http://stackoverflow.com/questions/18501081/generating-random-number-within-cuda-kernel-in-a-varying-range
    // FIXME: This isn't hitting 99,9999???
    float myrandf = curand_uniform(&states[tid]);
    myrandf *= ((float)(N[0]) - 0.0+0.9999999999999999);
    myrandf += 0.0;
    int city_one_swap = (int)truncf(myrandf);


    // This is the maximum we can sample from
    int sample_space = (int)floor(exp(-1 / T[0]) * (float)N[0]);
    // We need to set the min and max of the second city swap
    int min_city_two = (city_one_swap - sample_space > 0)?
        city_one_swap - sample_space:
           1;

    int max_city_two = (city_one_swap + sample_space < N[0])?
        city_one_swap + sample_space:
            (N[0] - 1);
    myrandf = curand_uniform(&states[tid]);
    myrandf *= ((float)max_city_two - (float)min_city_two + 0.999999999999999);
    myrandf += min_city_two;
    int city_two_swap = (int)truncf(myrandf);

    // This shouldn't have to be here, but if either is larger or equal to N
    // We set it to N[0] - 1
    if (city_one_swap >= N[0])
        city_one_swap = (N[0] - 1);
    if (city_two_swap >= N[0])
        city_two_swap = (N[0] - 1);
    city_one[tid] = city_one_swap;
    city_two[tid] = city_two_swap;

    float delta, p;
    unsigned int trip_city_one = salesman_route[city_one_swap];

    // NOTE: We set city two so that it could not be equal to 0 or N
    unsigned int trip_city_two      = salesman_route[city_two_swap];
    unsigned int trip_city_two_pre  = salesman_route[city_two_swap - 1];
    unsigned int trip_city_two_post = salesman_route[city_two_swap + 1];
    float original_dist = 0;
    float proposal_dist = 0;
    // We need to account for if city swap one is equal to 0
    if (city_one_swap != 0){
      unsigned int trip_city_one_pre  = salesman_route[city_one_swap - 1];
      unsigned int trip_city_one_post = salesman_route[city_one_swap + 1];

      // We will always have 4 calculations for original distance and the proposed distance
      // so we just unroll the loop here
      // TODO: It may be nice to make vars for the locations as well so this does not look so gross
      // The first city, unswapped. The one behind it and the one in front of it
      original_dist += (location[trip_city_one_pre].x - location[trip_city_one].x) *
                       (location[trip_city_one_pre].x - location[trip_city_one].x) +
                       (location[trip_city_one_pre].y - location[trip_city_one].y) *
                       (location[trip_city_one_pre].y - location[trip_city_one].y);
      original_dist += (location[trip_city_one_post].x - location[trip_city_one].x) *
                       (location[trip_city_one_post].x - location[trip_city_one].x) +
                       (location[trip_city_one_post].y - location[trip_city_one].y) *
                       (location[trip_city_one_post].y - location[trip_city_one].y);
      // The second city, unswapped. The one behind it and the one in front of it
      original_dist += (location[trip_city_two_pre].x - location[trip_city_two].x) *
                       (location[trip_city_two_pre].x - location[trip_city_two].x) +
                       (location[trip_city_two_pre].y - location[trip_city_two].y) *
                       (location[trip_city_two_pre].y - location[trip_city_two].y);
      original_dist += (location[trip_city_two_post].x - location[trip_city_two].x) *
                       (location[trip_city_two_post].x - location[trip_city_two].x) +
                       (location[trip_city_two_post].y - location[trip_city_two].y) *
                       (location[trip_city_two_post].y - location[trip_city_two].y);
      // The first city, swapped. The one behind it and the one in front of it
      proposal_dist += (location[trip_city_two_pre].x - location[trip_city_one].x) *
                       (location[trip_city_two_pre].x - location[trip_city_one].x) +
                       (location[trip_city_two_pre].y - location[trip_city_one].y) *
                       (location[trip_city_two_pre].y - location[trip_city_one].y);
      proposal_dist += (location[trip_city_two_post].x - location[trip_city_one].x) *
                       (location[trip_city_two_post].x - location[trip_city_one].x) +
                       (location[trip_city_two_post].y - location[trip_city_one].y) *
                       (location[trip_city_two_post].y - location[trip_city_one].y);
      // The second city, swapped. The one behind it and the one in front of it
      proposal_dist += (location[trip_city_one_pre].x - location[trip_city_two].x) *
                       (location[trip_city_one_pre].x - location[trip_city_two].x) +
                       (location[trip_city_one_pre].y - location[trip_city_two].y) *
                       (location[trip_city_one_pre].y - location[trip_city_two].y);
      proposal_dist += (location[trip_city_one_post].x - location[trip_city_two].x) *
                       (location[trip_city_one_post].x - location[trip_city_two].x) +
                       (location[trip_city_one_post].y - location[trip_city_two].y) *
                       (location[trip_city_one_post].y - location[trip_city_two].y);
    } else {
    // Now if zero is select we check the distance of the starting point and end point
      unsigned int trip_city_one_start  = salesman_route[1];
      unsigned int trip_city_one_end    = salesman_route[(N[0]-1)];


      original_dist += (location[trip_city_one_start].x - location[trip_city_one].x) *
                       (location[trip_city_one_start].x - location[trip_city_one].x) +
                       (location[trip_city_one_start].y - location[trip_city_one].y) *
                       (location[trip_city_one_start].y - location[trip_city_one].y);
      original_dist += (location[trip_city_one_end].x - location[trip_city_one].x) *
                       (location[trip_city_one_end].x - location[trip_city_one].x) +
                       (location[trip_city_one_end].y - location[trip_city_one].y) *
                       (location[trip_city_one_end].y - location[trip_city_one].y);
      // The second city, unswapped. The one behind it and the one in front of it
      original_dist += (location[trip_city_two_pre].x - location[trip_city_two].x) *
                       (location[trip_city_two_pre].x - location[trip_city_two].x) +
                       (location[trip_city_two_pre].y - location[trip_city_two].y) *
                       (location[trip_city_two_pre].y - location[trip_city_two].y);
      original_dist += (location[trip_city_two_post].x - location[trip_city_two].x) *
                       (location[trip_city_two_post].x - location[trip_city_two].x) +
                       (location[trip_city_two_post].y - location[trip_city_two].y) *
                       (location[trip_city_two_post].y - location[trip_city_two].y);
      // The first city, swapped. The one behind it and the one in front of it
      proposal_dist += (location[trip_city_two_pre].x - location[trip_city_one].x) *
                       (location[trip_city_two_pre].x - location[trip_city_one].x) +
                       (location[trip_city_two_pre].y - location[trip_city_one].y) *
                       (location[trip_city_two_pre].y - location[trip_city_one].y);
      proposal_dist += (location[trip_city_two_post].x - location[trip_city_one].x) *
                       (location[trip_city_two_post].x - location[trip_city_one].x) +
                       (location[trip_city_two_post].y - location[trip_city_one].y) *
                       (location[trip_city_two_post].y - location[trip_city_one].y);
      // The second city, swapped. The one behind it and the one in front of it
      proposal_dist += (location[trip_city_one_start].x - location[trip_city_two].x) *
                       (location[trip_city_one_start].x - location[trip_city_two].x) +
                       (location[trip_city_one_start].y - location[trip_city_two].y) *
                       (location[trip_city_one_start].y - location[trip_city_two].y);
      proposal_dist += (location[trip_city_one_end].x - location[trip_city_two].x) *
                       (location[trip_city_one_end].x - location[trip_city_two].x) +
                       (location[trip_city_one_end].y - location[trip_city_two].y) *
                       (location[trip_city_one_end].y - location[trip_city_two].y);
    }

    //picking the first accepted and picking the last accepted is equivalent, and here I pick the latter one
    //because if I pick the small one, I have to tell whether the flag is 0
    if (proposal_dist < original_dist&&global_flag[0]<tid){
        global_flag[0] = tid;
        __syncthreads();
    } else {
        delta = proposal_dist - original_dist;
        p = exp(-delta / T[0]);
        myrandf = curand_uniform(&states[tid]);
        if (p > myrandf && global_flag[0]<tid){
            global_flag[0] = tid;
            __syncthreads();
        }
    }
    iter++;
    }
    //seed[tid] = r_r;   //refresh the seed at the end of kernel
}

#endif // _TSP_SOLVE_H_
