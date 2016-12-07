#ifndef _TSP_SWAP_H_
#define _TSP_SWAP_H_



/* TSP Using the city swap method
Input:
- city_one: [unsigned integer(GRID_SIZE)]
 > Cities to swap for the first swap choice
- city_two: [unsigned integer(GRID_SIZE)]
 > Cities to swap for the second swap choice
- location [coordinate(N)]
 > The struct that holds the x and y coordinate information
- salesman_route: [unsigned integer(N + 1)]
 > The route the salesman will travel, starting and ending in the same position
- T: [unsigned integer(1)]
 > The current temperature
- N [unsigned integer(1)]
 > The number of cities.
- states [curandState_t(GRID_SIZE)]
 > The seeds for each proposal steps random sample
*/


__global__ static void globalSwap(unsigned int* city_one,
                           unsigned int* city_two,
                           coordinates* __restrict__ location,
                           unsigned int* __restrict__ salesman_route,
                           float* __restrict__ T,
                           volatile unsigned int *global_flag,
                           unsigned int* __restrict__ N,
                           curandState_t* states){

    //first, refresh the route, this time we have to change city_one-city_two elements
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int iter = 0;
    // This is the maximum we can sample from
    int sample_space = (int)floor(exp(- (T[1] / 15) / T[0]) * N[0] + 3);
    while (global_flag[0] == 0 && iter < 3){



        // Generate the first city
        // From: http://stackoverflow.com/questions/18501081/generating-random-number-within-cuda-kernel-in-a-varying-range
        float myrandf = curand_uniform(&states[tid]);
        myrandf *= ((float)(N[0] - 4) - 4.0+0.9999999999999999);
        myrandf += 4.0;
        //myrandf += (curand_normal(&states[tid]) * sample_space);
        // Do uniform for city one, and normal dist for city two
        int city_one_swap = (int)truncf(myrandf);
        //if (city_one_swap >= N[0]) city_one_swap -= (city_one_swap/N[0] + 1) * N[0];
        //if (city_one_swap < 0)    city_one_swap += (-city_one_swap/N[0] + 1) * N[0];

        myrandf = city_one_swap + (curand_normal(&states[tid]) * sample_space);
        int city_two_swap = (int)truncf(myrandf);
        // This wheels city two around the circle
        if (city_two_swap >= N[0]) city_two_swap -= (city_two_swap/N[0]) * N[0] + 1;
        // We add 1 here because we want to shift up N form [-N[0] to 0]
        if (city_two_swap <= 0)    city_two_swap += (-city_two_swap/N[0] + 1) * N[0] - 1;

        // Check if city one is too close to city two.
        if ( (city_two_swap - city_one_swap) * (city_two_swap - city_one_swap) < 4){
         // If less, shift down 3
           if (city_two_swap < city_one_swap) city_two_swap = city_one_swap - 1;
        // If more, shift up 3
           if (city_two_swap > city_one_swap) city_two_swap = city_one_swap+ 1;
        }


    city_one[tid] = city_one_swap;
    city_two[tid] = city_two_swap;

    float quotient, p;
    // Set the swap cities in the trip.
    unsigned int trip_city_one = salesman_route[city_one_swap];
    unsigned int trip_city_one_pre  = salesman_route[city_one_swap - 1];
    unsigned int trip_city_one_post = salesman_route[city_one_swap + 1];
    unsigned int trip_city_two      = salesman_route[city_two_swap];
    unsigned int trip_city_two_pre  = salesman_route[city_two_swap - 1];
    unsigned int trip_city_two_post = salesman_route[city_two_swap + 1];
    float original_dist = 0;
    float proposal_dist = 0;



      // We will always have 4 calculations for original distance and the proposed distance
      // so we just unroll the loop here
      // TODO: It may be nice to make vars for the locations as well so this does not look so gross
      // The first city, unswapped. The one behind it and the one in front of it
      original_dist += sqrtf(powf(location[trip_city_one_pre].x - location[trip_city_one].x,2)  +
                       powf(location[trip_city_one_pre].y - location[trip_city_one].y,2));
      original_dist += sqrtf(powf(location[trip_city_one_post].x - location[trip_city_one].x,2) +
                       powf(location[trip_city_one_post].y - location[trip_city_one].y,2));
      // The second city, unswapped. The one behind it and the one in front of it
      original_dist += sqrtf(powf(location[trip_city_two_pre].x - location[trip_city_two].x,2) +
                       powf(location[trip_city_two_pre].y - location[trip_city_two].y,2));
      original_dist += sqrtf(powf(location[trip_city_two_post].x - location[trip_city_two].x,2) +
                       powf(location[trip_city_two_post].y - location[trip_city_two].y,2));
      // The first city, swapped. The one behind it and the one in front of it
      proposal_dist += sqrtf(powf(location[trip_city_two_pre].x - location[trip_city_one].x,2)  +
                       powf(location[trip_city_two_pre].y - location[trip_city_one].y,2));
      proposal_dist += sqrtf(powf(location[trip_city_two_post].x - location[trip_city_one].x,2) +
                       powf(location[trip_city_two_post].y - location[trip_city_one].y,2));
      // The second city, swapped. The one behind it and the one in front of it
      proposal_dist += sqrtf(powf(location[trip_city_one_pre].x - location[trip_city_two].x,2)  +
                       powf(location[trip_city_one_pre].y - location[trip_city_two].y,2));
      proposal_dist += sqrtf(powf(location[trip_city_one_post].x - location[trip_city_two].x,2)  +
                       powf(location[trip_city_one_post].y - location[trip_city_two].y,2));


    //picking the first accepted and picking the last accepted is equivalent, and here I pick the latter one
    //because if I pick the small one, I have to tell whether the flag is 0
    if (proposal_dist < original_dist && global_flag[0] == 0){
        global_flag[0] = tid;
        __threadfence();
    }
    if (T[0] > 1){
    if (global_flag[0]==0){
        quotient = proposal_dist / original_dist - 1;
        // You can change the constant to whatever you would like
        // But you should check that the graph looks nice
        //http://www.wolframalpha.com/input/?i=e%5E(-(x*(10000%2F5))%2Ft)+x+%3D+0+to+3+and+t+%3D+0+to+10000
        p = exp(-(quotient * T[1] * 100) / T[0]);
        myrandf = curand_uniform(&states[tid]);
        if (p > myrandf && global_flag[0]<tid){
            global_flag[0] = tid;
            __syncthreads();
        }
     }
     }
    iter++;
    }
    //seed[tid] = r_r;   //refresh the seed at the end of kernel
}

__global__ static void localSwap(unsigned int* city_one,
    unsigned int* city_two,
    coordinates* __restrict__ location,
    unsigned int* __restrict__ salesman_route,
    float* __restrict__ T,
    volatile unsigned int *global_flag,
    unsigned int* __restrict__ N,
    curandState_t* states){

    //first, refresh the route, this time we have to change city_one-city_two elements
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int iter = 0;
    // This is the maximum we can sample from
    int sample_space = (int)floor(exp(- (T[1] / 15) / T[0]) * N[0] + 3);
    while (global_flag[0] == 0 && iter < 3){



        // Generate the first city
        // From: http://stackoverflow.com/questions/18501081/generating-random-number-within-cuda-kernel-in-a-varying-range
        float myrandf = curand_uniform(&states[tid]);
        myrandf *= ((float)(N[0] - 4) - 4.0+0.9999999999999999);
        myrandf += 4.0;
        //myrandf += (curand_normal(&states[tid]) * sample_space);
        // Do uniform for city one, and normal dist for city two
        int city_one_swap = (int)truncf(myrandf);
        //if (city_one_swap >= N[0]) city_one_swap -= (city_one_swap/N[0] + 1) * N[0];
        //if (city_one_swap < 0)    city_one_swap += (-city_one_swap/N[0] + 1) * N[0];

        myrandf = city_one_swap + (curand_normal(&states[tid]) * sample_space);
        int city_two_swap = (int)truncf(myrandf);
        // This wheels city two around the circle
        if (city_two_swap >= N[0]) city_two_swap -= (city_two_swap/N[0]) * N[0] + 1;
        if (city_two_swap <= 0)    city_two_swap += (-city_two_swap/N[0] + 1) * N[0] - 1;

        // Check if city one is too close to city two.
        if ( (city_two_swap - city_one_swap) * (city_two_swap - city_one_swap) < 4){
         // If less, shift down 3
           if (city_two_swap < city_one_swap) city_two_swap = city_one_swap - 1;
        // If more, shift up 3
           if (city_two_swap > city_one_swap) city_two_swap = city_one_swap+ 1;
        }


    city_one[tid] = city_one_swap;
    city_two[tid] = city_two_swap;

    float quotient, p;
    // Set the swap cities in the trip.
    unsigned int trip_city_one = salesman_route[city_one_swap];
    unsigned int trip_city_one_pre  = salesman_route[city_one_swap - 1];
    unsigned int trip_city_one_post = salesman_route[city_one_swap + 1];
    unsigned int trip_city_two      = salesman_route[city_two_swap];
    unsigned int trip_city_two_pre  = salesman_route[city_two_swap - 1];
    unsigned int trip_city_two_post = salesman_route[city_two_swap + 1];
    float original_dist = 0;
    float proposal_dist = 0;



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


    //picking the first accepted and picking the last accepted is equivalent, and here I pick the latter one
    //because if I pick the small one, I have to tell whether the flag is 0
    if (proposal_dist < original_dist && global_flag[0] == 0){
        global_flag[0] = tid;
        __threadfence();
    }
    if (T[0] > 1){
    if (global_flag[0]==0){
        quotient = proposal_dist / original_dist - 1;
        // You can change the constant to whatever you would like
        // But you should check that the graph looks nice
        //http://www.wolframalpha.com/input/?i=e%5E(-(x*(10000%2F5))%2Ft)+x+%3D+0+to+3+and+t+%3D+0+to+10000
        p = exp(-(quotient * T[1] * 100) / T[0]);
        myrandf = curand_uniform(&states[tid]);
        if (p > myrandf && global_flag[0]<tid){
            global_flag[0] = tid;
            __syncthreads();
        }
     }
     }
    iter++;
    }
    //seed[tid] = r_r;   //refresh the seed at the end of kernel
}

__global__ static void SwapUpdate(unsigned int* __restrict__ city_one,
                           unsigned int* __restrict__ city_two,
                           unsigned int* __restrict__ salesman_route,
                           volatile unsigned int *global_flag){

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tmp;
    // use thread 0 to refresh the route
    if (tid == 0){
        if (global_flag[0] != 0){
            tmp = salesman_route[city_one[global_flag[0]]];
            salesman_route[city_one[global_flag[0]]] = salesman_route[city_two[global_flag[0]]];
            salesman_route[city_two[global_flag[0]]] = tmp;
            global_flag[0] = 0;
        }
    }
    __syncthreads();
    __threadfence();
}

#endif // _TSP_SWAP_H_

