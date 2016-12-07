#ifndef _TSP_2OPT_H_
#define _TSP_2OPT_H_

/* TSP Using the 2-opt method
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


__global__ static void global2Opt(unsigned int* city_one,
	unsigned int* city_two,
	coordinates* __restrict__ location,
	unsigned int* __restrict__ salesman_route,
	float* __restrict__ T,
	volatile unsigned int *global_flag,
	unsigned int* __restrict__ N,
	curandState_t* states){

	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid == 0)
		global_flag[0] = 0;
	int iter = 0;
	//insertion and swap all decrease to 1 at last, so I set it a little larger,30
	int sample_space = (int)floor(5 + exp(- (T[1] / 15) / T[0]) * (float)N[0]);
	while (global_flag[0] == 0 && iter < 3){
        //the first city's indice has to be smaller than the second, to simplify the algo
		float myrandf = curand_uniform(&states[tid]);
		myrandf *= ((float)(N[0] - 5) - 1.0 + 0.9999999999999999);
		myrandf += 1.0;
		int city_one_swap = (int)truncf(myrandf);
        if (city_one_swap > N[0]) city_one_swap -= (city_one_swap/N[0]) * N[0] - 5;
        if (city_one_swap <= 0) city_one_swap += -(city_one_swap/N[0] + 1) * N[0] + 3;
		// +1,wrong;+2,equivalent to swap; so start with +3
		int min_city_two = city_one_swap+3;

		int max_city_two = (city_one_swap + 3 + sample_space < N[0]-1) ?
			city_one_swap+ 3 + sample_space :
			(N[0] - 1);
		myrandf = curand_uniform(&states[tid]);
		myrandf *= ((float)max_city_two - (float)min_city_two + 0.999999999999999);
		myrandf += min_city_two;
		int city_two_swap = (int)truncf(myrandf);

		city_one[tid] = city_one_swap;
		city_two[tid] = city_two_swap;

		float quotient, p;
		// Set the swap cities in the trip.
		unsigned int trip_city_one = salesman_route[city_one_swap];
		unsigned int trip_city_one_post = salesman_route[city_one_swap + 1];
		unsigned int trip_city_two = salesman_route[city_two_swap];
		unsigned int trip_city_two_post = salesman_route[city_two_swap + 1];
		float original_dist = 0;
		float proposal_dist = 0;



		// very simple relationship
		original_dist += sqrtf(powf(location[trip_city_one_post].x - location[trip_city_one].x,2) +
			             powf(location[trip_city_one_post].y - location[trip_city_one].y,2));
		original_dist += sqrtf(powf(location[trip_city_two_post].x - location[trip_city_two].x ,2) +
			             powf(location[trip_city_two_post].y - location[trip_city_two].y,2));
			             
		proposal_dist += sqrtf(powf(location[trip_city_two].x - location[trip_city_one].x,2) +
			             powf(location[trip_city_two].y - location[trip_city_one].y,2));
		proposal_dist += sqrtf(powf(location[trip_city_two_post].x - location[trip_city_one_post].x,2) +
			             powf(location[trip_city_two_post].y - location[trip_city_one_post].y,2));


		//I think if we have three methods, there's no need for acceptance...
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

__global__ static void Opt2Update(unsigned int* __restrict__ city_one,
	unsigned int* __restrict__ city_two,
	unsigned int* salesman_route,
	unsigned int* salesman_route2,
	volatile unsigned int *global_flag){

	// each thread is a position in the salesman's trip
	const int xid = blockIdx.x * blockDim.x + threadIdx.x;
	/*
	remember to refresh route2 before calling this function, just like insertion
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
