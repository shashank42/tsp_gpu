//kernel part
__global__ static void tsp_2_Opt(unsigned int* city_one,
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
	while (iter < 100){
        //the first city's indice has to be smaller than the second, to simplify the algo
		float myrandf = curand_uniform(&states[tid]);
		myrandf *= ((float)(N[0] - 33) - 1.0 + 0.9999999999999999);
		myrandf += 1.0;
		int city_one_swap = (int)truncf(myrandf);
        //insertion and swap all decrease to 1 at last, so I set it a little larger,30
		int sample_space = (int)floor(30 + exp(-0.01 / T[0]) * (float)N[0]);
		// +1,wrong;+2,equivalent to swap; so start with +3
		int min_city_two = city_one_swap+3;

		int max_city_two = (city_one_swap + sample_space < N[0]-1) ?
			city_one_swap+3 + sample_space :
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



		// very simple relationship
		original_dist += (location[trip_city_one_post].x - location[trip_city_one].x) *
			(location[trip_city_one_post].x - location[trip_city_one].x) +
			(location[trip_city_one_post].y - location[trip_city_one].y) *
			(location[trip_city_one_post].y - location[trip_city_one].y);
		original_dist += (location[trip_city_two_post].x - location[trip_city_two].x) *
			(location[trip_city_two_post].x - location[trip_city_two].x) +
			(location[trip_city_two_post].y - location[trip_city_two].y) *
			(location[trip_city_two_post].y - location[trip_city_two].y);
		proposal_dist += (location[trip_city_two].x - location[trip_city_one].x) *
			(location[trip_city_two].x - location[trip_city_one].x) +
			(location[trip_city_two].y - location[trip_city_one].y) *
			(location[trip_city_two].y - location[trip_city_one].y);
		proposal_dist += (location[trip_city_two_post].x - location[trip_city_one_post].x) *
			(location[trip_city_two_post].x - location[trip_city_one_post].x) +
			(location[trip_city_two_post].y - location[trip_city_one_post].y) *
			(location[trip_city_two_post].y - location[trip_city_one_post].y);


		//I think if we have three methods, there's no need for acceptance...
		if (proposal_dist < original_dist&&global_flag[0]<tid){
			global_flag[0] = tid;
			__syncthreads();
		}
		iter++;
		if (global_flag[0] != 0)
			iter += 100;
	}
	//seed[tid] = r_r;   //refresh the seed at the end of kernel
}

__global__ static void tsp_2_Opt_Update(unsigned int* __restrict__ city_one,
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

//host part, just like insertion

tsp_2_Opt << <blocksPerSampleGrid, threadsPerBlock, 0 >> >(city_swap_one_g, city_swap_two_g,
				location_g, salesman_route_g,
				T_g, global_flag_g, N_g,
				states);
			cudaCheckError();
			tspInsertionUpdateTrip << <blocksPerTripGrid, threadsPerBlock, 0 >> >(salesman_route_g, salesman_route_2g, N_g);
			cudaCheckError();
			tsp_2_Opt_Update << <blocksPerTripGrid, threadsPerBlock, 0 >> >(city_swap_one_g, city_swap_two_g,
				salesman_route_g, salesman_route_2g, global_flag_g);
			cudaCheckError();



