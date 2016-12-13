#ifndef _TSP_2OPT_H_
#define _TSP_2OPT_H_


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

__global__ static void twoOptStep(unsigned int* city_one,
	unsigned int* city_two,
	coordinates* __restrict__ location,
	unsigned int* __restrict__ salesman_route,
	float* __restrict__ T,
	volatile unsigned int *global_flag,
	unsigned int* __restrict__ N,
	curandState_t* states,
	float* sample_area){

	const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid == 0)
		global_flag[0] = 0;
	int iter = 0;
	// This is the maximum we can sample from
	// This gives us a nice curve
	//http://www.wolframalpha.com/input/?i=e%5E(-+.2%2Ft)+from+0+to+1
	int sample_space = (int)floor(30 + exp(-sample_area[0] / T[0]) * (float)N[0]);
	// Run until either global flag is zero and we do 100 iterations is false.
	while (iter < 10){
		// Generate the first city
		// From: http://stackoverflow.com/questions/18501081/generating-random-number-within-cuda-kernel-in-a-varying-range
		// FIXME: This isn't hitting 99,9999???
		float myrandf = curand_uniform(&states[tid]);
		myrandf *= ((float)(N[0] - 4) - 1.0 + 0.9999999999999999);
		myrandf += 1.0;
		int city_one_swap = (int)truncf(myrandf);




		// We need to set the min and max of the second city swap
		int min_city_two = city_one_swap+2;

		int max_city_two = (city_one_swap + sample_space +2 < N[0]-1) ?
			city_one_swap+2 + sample_space :
			(N[0] - 2);
		myrandf = curand_uniform(&states[tid]);
		myrandf *= ((float)max_city_two - (float)min_city_two + 0.999999999999999);
		myrandf += min_city_two;
		int city_two_swap = (int)truncf(myrandf);

		// This shouldn't have to be here, but if either is larger or equal to N
		// We set it to N[0] - 1
		//I think since the space is well setted, this is innecessary
	//	if (city_one_swap >= N[0])
	//		city_one_swap = (N[0] - 1);
//		if (city_two_swap >= N[0])
	//		city_two_swap = (N[0] - 1);

		city_one[tid] = city_one_swap;
		city_two[tid] = city_two_swap;

	//	float quotient, p;
		// Set the swap cities in the trip.
		unsigned int trip_city_one = salesman_route[city_one_swap];
		unsigned int trip_city_one_post = salesman_route[city_one_swap + 1];
		unsigned int trip_city_two = salesman_route[city_two_swap];
		unsigned int trip_city_two_post = salesman_route[city_two_swap + 1];
		float original_dist = 0;
		float proposal_dist = 0;



		// We will always have 4 calculations for original distance and the proposed distance
		// so we just unroll the loop here
		// TODO: It may be nice to make vars for the locations as well so this does not look so gross
		// The first city, unswapped. The one behind it and the one in front of i
		original_dist += sqrtf((location[trip_city_one_post].x - location[trip_city_one].x) *
			(location[trip_city_one_post].x - location[trip_city_one].x) +
			(location[trip_city_one_post].y - location[trip_city_one].y) *
			(location[trip_city_one_post].y - location[trip_city_one].y));
		// The second city, unswapped. The one behind it and the one in front of it
		original_dist += sqrtf((location[trip_city_two_post].x - location[trip_city_two].x) *
			(location[trip_city_two_post].x - location[trip_city_two].x) +
			(location[trip_city_two_post].y - location[trip_city_two].y) *
			(location[trip_city_two_post].y - location[trip_city_two].y));
		// The first city, swapped. The one behind it and the one in front of it
		proposal_dist += sqrtf((location[trip_city_two].x - location[trip_city_one].x) *
			(location[trip_city_two].x - location[trip_city_one].x) +
			(location[trip_city_two].y - location[trip_city_one].y) *
			(location[trip_city_two].y - location[trip_city_one].y));
		proposal_dist += sqrtf((location[trip_city_two_post].x - location[trip_city_one_post].x) *
			(location[trip_city_two_post].x - location[trip_city_one_post].x) +
			(location[trip_city_two_post].y - location[trip_city_one_post].y) *
			(location[trip_city_two_post].y - location[trip_city_one_post].y));


		//picking the first accepted and picking the last accepted is equivalent, and here I pick the latter one
		//because if I pick the small one, I have to tell whether the flag is 0
		if (proposal_dist < original_dist&&global_flag[0]<tid){
			global_flag[0] = tid;
			__threadfence();
		}
		iter++;
		if (global_flag[0] != 0)
			iter += 100;
	}
	//seed[tid] = r_r;   //refresh the seed at the end of kernel
}

__global__ static void opt2Update(unsigned int* __restrict__ city_one,
	unsigned int* __restrict__ city_two,
	unsigned int* salesman_route,
	unsigned int* salesman_route2,
	volatile unsigned int *global_flag){

	// each thread is a position in the salesman's trip
	const int xid = blockIdx.x * blockDim.x + threadIdx.x;
	/*
	1. Save city one
	2. Shift everything between city one and city two up or down, depending on city one < city two
	3. Set city two's old position to city one
	*/
	if (global_flag[0] != 0){
		unsigned int city_one_swap = city_one[global_flag[0]];
		unsigned int city_two_swap = city_two[global_flag[0]];

		if (xid > city_one_swap && xid <= city_two_swap){
			salesman_route[xid] = salesman_route2[city_one_swap+city_two_swap+1-xid];
		}
	}
}


#endif // _TSP_2OPT_H_

