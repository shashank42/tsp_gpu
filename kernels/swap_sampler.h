/*
    columbus: Software for computing approximate solutions to the traveling salesman's problem on GPUs
    Copyright (C) 2016 Steve Bronder and Haoyan Min

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

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

__global__ static void swapStep(unsigned int* city_one,
                           unsigned int* city_two,
                           coordinates* __restrict__ location,
                           unsigned int* __restrict__ salesman_route,
                           float* __restrict__ T,
                           volatile int *global_flag,
                           unsigned int* __restrict__ N,
                           curandState_t* states,
                           float * sample_area){

    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int iter = 0;
	if (tid == 0)
		global_flag[0] = -1;
    // Run until either global flag is zero and we do 100 iterations is false.
    // This is the maximum we can sample from
    // This gives us a nice curve
    //http://www.wolframalpha.com/input/?i=e%5E(-+.2%2Ft)+from+0+to+1
    int sample_space = (int)floor(30+exp(- sample_area[0] / T[0]) * (float)N[0]);
    while (iter < 5){
    // Generate the first city
    // From: http://stackoverflow.com/questions/18501081/generating-random-number-within-cuda-kernel-in-a-varying-range
    float myrandf = curand_uniform(&states[tid]);
    myrandf *= ((float)(N[0] - 4) - 4.0+0.9999999999999999);
    myrandf += 4.0;
    int city_one_swap = (int)truncf(myrandf);

    if (city_one_swap > N[0] - 4) city_one_swap = N[0] - 4;
    if (city_one_swap < 4) city_one_swap = 4;
    
    // Trying out normally distributed swap step
    int city_two_swap = (int)(city_one_swap + (curand_normal(&states[tid]) * sample_space));
    
    // One is added here so that if we have city two == N, then it bumps it up to 1
    if (city_two_swap >= N[0]) city_two_swap -= (city_two_swap/N[0]) * N[0] + 1;
    if (city_two_swap <= 0) city_two_swap += (-city_two_swap/N[0] + 1) * N[0] - 1;
    if (city_two_swap > N[0] - 1) city_two_swap = N[0] - 1;
    if (city_two_swap < 1) city_two_swap = 1;
    // Check it ||city_two - city one|| < 9, if so bump it up one
    if ((city_one_swap - city_two_swap) * (city_one_swap - city_two_swap) < 9)
        city_two_swap = city_one_swap + 3;
/*
    // We need to set the min and max of the second city swap
	int min_city_two = city_one_swap + 3;
    int max_city_two = (city_one_swap +3+ sample_space < N[0])?
        city_one_swap +3+ sample_space:
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
*/
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
	original_dist += sqrtf((location[trip_city_one_pre].x - location[trip_city_one].x) *
		(location[trip_city_one_pre].x - location[trip_city_one].x) +
		(location[trip_city_one_pre].y - location[trip_city_one].y) *
		(location[trip_city_one_pre].y - location[trip_city_one].y));
	original_dist += sqrtf((location[trip_city_one_post].x - location[trip_city_one].x) *
		(location[trip_city_one_post].x - location[trip_city_one].x) +
		(location[trip_city_one_post].y - location[trip_city_one].y) *
		(location[trip_city_one_post].y - location[trip_city_one].y));
	// The second city, unswapped. The one behind it and the one in front of it
	original_dist += sqrtf((location[trip_city_two_pre].x - location[trip_city_two].x) *
		(location[trip_city_two_pre].x - location[trip_city_two].x) +
		(location[trip_city_two_pre].y - location[trip_city_two].y) *
		(location[trip_city_two_pre].y - location[trip_city_two].y));
	original_dist += sqrtf((location[trip_city_two_post].x - location[trip_city_two].x) *
		(location[trip_city_two_post].x - location[trip_city_two].x) +
		(location[trip_city_two_post].y - location[trip_city_two].y) *
		(location[trip_city_two_post].y - location[trip_city_two].y));
	// The first city, swapped. The one behind it and the one in front of it
	proposal_dist += sqrtf((location[trip_city_two_pre].x - location[trip_city_one].x) *
		(location[trip_city_two_pre].x - location[trip_city_one].x) +
		(location[trip_city_two_pre].y - location[trip_city_one].y) *
		(location[trip_city_two_pre].y - location[trip_city_one].y));
	proposal_dist += sqrtf((location[trip_city_two_post].x - location[trip_city_one].x) *
		(location[trip_city_two_post].x - location[trip_city_one].x) +
		(location[trip_city_two_post].y - location[trip_city_one].y) *
		(location[trip_city_two_post].y - location[trip_city_one].y));
	// The second city, swapped. The one behind it and the one in front of it
	proposal_dist += sqrtf((location[trip_city_one_pre].x - location[trip_city_two].x) *
		(location[trip_city_one_pre].x - location[trip_city_two].x) +
		(location[trip_city_one_pre].y - location[trip_city_two].y) *
		(location[trip_city_one_pre].y - location[trip_city_two].y));
	proposal_dist += sqrtf((location[trip_city_one_post].x - location[trip_city_two].x) *
		(location[trip_city_one_post].x - location[trip_city_two].x) +
		(location[trip_city_one_post].y - location[trip_city_two].y) *
		(location[trip_city_one_post].y - location[trip_city_two].y));


    //picking the first accepted and picking the last accepted is equivalent, and here I pick the latter one
    //because if I pick the small one, I have to tell whether the flag is 0
	  if (proposal_dist < original_dist && global_flag[0] == -1){
		  global_flag[0] = tid;
		  __threadfence();
	  }
	  else if (global_flag[0] == -1)
	{
        quotient = proposal_dist/original_dist-1;
        p = exp(-quotient * 6000 / T[0]);
        myrandf = curand_uniform(&states[tid]);
        if (p > myrandf && global_flag[0] == -1){
            global_flag[0] = tid;
            __syncthreads();
        }
     }
    iter++;
	if (global_flag[0] != -1)
		iter += 100;
    }
    //seed[tid] = r_r;   //refresh the seed at the end of kernel
}


__global__ static void swapUpdate(unsigned int* __restrict__ city_one,
                           unsigned int* __restrict__ city_two,
                           unsigned int* __restrict__ salesman_route,
                           volatile int *global_flag){

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tmp;
    // use thread 0 to refresh the route
    if (tid == 0){
        if (global_flag[0] != -1){
            tmp = salesman_route[city_one[global_flag[0]]];
            salesman_route[city_one[global_flag[0]]] = salesman_route[city_two[global_flag[0]]];
            salesman_route[city_two[global_flag[0]]] = tmp;
            global_flag[0] = -1;
        }
    }
    __syncthreads();
}


#endif // _TSP_SWAP_H_

