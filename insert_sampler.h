#ifndef _TSP_INSERT_H_
#define _TSP_INSERT_H_


/* TSP using the insertion method
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
__global__ static void globalInsertion(unsigned int* city_one,
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
    while (global_flag[0] == 0 && iter < 50){
      // Generate the first city
    // From: http://stackoverflow.com/questions/18501081/generating-random-number-within-cuda-kernel-in-a-varying-range
    float myrandf = curand_uniform(&states[tid]);
    myrandf *= ((float)(N[0] - 1) - 1.0+0.9999999999999999);
    myrandf += 1.0;
    int city_one_swap = (int)truncf(myrandf);



    // This is the maximum we can sample from
    int sample_space = (int)floor(exp(- (T[1]/8) / T[0]) * N[0]);
    // We need to set the min and max of the second city swap
    int min_city_two = (city_one_swap - sample_space > 0)?
        city_one_swap - sample_space:
           1;

    int max_city_two = (city_one_swap + sample_space < N[0] - 1)?
        city_one_swap + sample_space:
            (N[0] - 2);
    myrandf = curand_uniform(&states[tid]);
    myrandf *= ((float)max_city_two - (float)min_city_two + 0.999999999999999);
    myrandf += min_city_two;
    int city_two_swap = (int)truncf(myrandf);

    // This shouldn't have to be here, but if either is larger or equal to N - 2
    // We set it to N[0] - 2
    if (city_one_swap > N[0] - 2)
        city_one_swap = (N[0] - 2);
    if (city_two_swap > N[0] - 2)
        city_two_swap = (N[0] - 2);
    // END NEW


    if (city_two_swap !=(N[0]-1) && city_two_swap!=city_one_swap && city_two_swap!=city_one_swap-1)
    {
        city_one[tid] = city_one_swap;
        city_two[tid] = city_two_swap;

        float quotient, p;
        unsigned int trip_city_one = salesman_route[city_one_swap];
        unsigned int trip_city_one_pre = salesman_route[city_one_swap - 1];
        unsigned int trip_city_one_post = salesman_route[city_one_swap + 1];

        unsigned int trip_city_two = salesman_route[city_two_swap];
        unsigned int trip_city_two_post = salesman_route[city_two_swap + 1];
        // The original and post distances
        float original_dist = 0;
        float proposal_dist = 0;

        /* City one is the city to be inserted between city two and city two + 1
           That means we only have to make three calculations to compute each loss funciton
           original:
             - city one - 1 -> city one
             - city one -> city one + 1
             - City two -> city two + 1
           proposal:
             - city two -> city one
             - city one -> city two + 1
             - city one - 1 -> city one + 1
        */
        original_dist += (location[trip_city_one_pre].x - location[trip_city_one].x) *
                         (location[trip_city_one_pre].x - location[trip_city_one].x) +
                         (location[trip_city_one_pre].y - location[trip_city_one].y) *
                         (location[trip_city_one_pre].y - location[trip_city_one].y);
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
        proposal_dist += (location[trip_city_two_post].x - location[trip_city_one].x) *
                         (location[trip_city_two_post].x - location[trip_city_one].x) +
                         (location[trip_city_two_post].y - location[trip_city_one].y) *
                         (location[trip_city_two_post].y - location[trip_city_one].y);
        proposal_dist += (location[trip_city_one_pre].x - location[trip_city_one_post].x) *
                         (location[trip_city_one_pre].x - location[trip_city_one_post].x) +
                         (location[trip_city_one_pre].y - location[trip_city_one_post].y) *
                         (location[trip_city_one_pre].y - location[trip_city_one_post].y);
        //picking the first accepted and picking the last accepted is equivalent, and here I pick the latter one
        //because if I pick the small one, I have to tell whether the flag is 0
        if (proposal_dist < original_dist&&global_flag[0]<tid){
            global_flag[0] = tid;
            
        }
        __syncthreads();
        if (global_flag[0]==0) {
        
            quotient = proposal_dist - original_dist; 
            p = exp(-quotient / T[0]);
            myrandf = curand_uniform(&states[tid]);
            if (p > myrandf && global_flag[0]<tid){ 
                global_flag[0] = tid;
                __syncthreads(); 
            }
        }
        iter++;
        } // End of while
  } 
}

__global__ static void localInsertion(unsigned int* city_one,
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

    while (global_flag[0] == 0 && iter < 50){
	// Generate the first city
	// From: http://stackoverflow.com/questions/18501081/generating-random-number-within-cuda-kernel-in-a-varying-range
	float myrandf = curand_uniform(&states[tid]);
	myrandf *= ((float)(N[0] - 1) - 1.0 + 0.9999999999999999);
	myrandf += 1.0;
	int city_one_swap = (int)truncf(myrandf);



	// This is the maximum we can sample from
	int sample_space = (int)floor(exp(- (T[1]/2) / T[0]) * N[0]);
	// We need to set the min and max of the second city swap
	int min_city_two = (city_one_swap - sample_space > 0) ?
		city_one_swap - sample_space :
		1;

	int max_city_two = (city_one_swap + sample_space < N[0] - 1) ?
		city_one_swap + sample_space :
		(N[0] - 2);
	myrandf = curand_uniform(&states[tid]);
	myrandf = 0.8 + myrandf*0.2;
	myrandf *= ((float)max_city_two - (float)min_city_two + 0.999999999999999);
	myrandf += min_city_two;
	int city_two_swap = (int)truncf(myrandf);

	// This shouldn't have to be here, but if either is larger or equal to N - 2
	// We set it to N[0] - 2
	if (city_one_swap > N[0] - 2)
		city_one_swap = (N[0] - 2);
	if (city_two_swap > N[0] - 2)
		city_two_swap = (N[0] - 2);
	// END NEW


	if (city_two_swap != (N[0] - 1) && city_two_swap != city_one_swap && city_two_swap != city_one_swap - 1)
	{
		city_one[tid] = city_one_swap;
		city_two[tid] = city_two_swap;

		float quotient, p;
		unsigned int trip_city_one = salesman_route[city_one_swap];
		unsigned int trip_city_one_pre = salesman_route[city_one_swap - 1];
		unsigned int trip_city_one_post = salesman_route[city_one_swap + 1];

		unsigned int trip_city_two = salesman_route[city_two_swap];
		unsigned int trip_city_two_post = salesman_route[city_two_swap + 1];
		// The original and post distances
		float original_dist = 0;
		float proposal_dist = 0;

		/* City one is the city to be inserted between city two and city two + 1
		That means we only have to make three calculations to compute each loss funciton
		original:
		- city one - 1 -> city one
		- city one -> city one + 1
		- City two -> city two + 1
		proposal:
		- city two -> city one
		- city one -> city two + 1
		- city one - 1 -> city one + 1
		*/
		original_dist += (location[trip_city_one_pre].x - location[trip_city_one].x) *
			(location[trip_city_one_pre].x - location[trip_city_one].x) +
			(location[trip_city_one_pre].y - location[trip_city_one].y) *
			(location[trip_city_one_pre].y - location[trip_city_one].y);
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
		proposal_dist += (location[trip_city_two_post].x - location[trip_city_one].x) *
			(location[trip_city_two_post].x - location[trip_city_one].x) +
			(location[trip_city_two_post].y - location[trip_city_one].y) *
			(location[trip_city_two_post].y - location[trip_city_one].y);
		proposal_dist += (location[trip_city_one_pre].x - location[trip_city_one_post].x) *
			(location[trip_city_one_pre].x - location[trip_city_one_post].x) +
			(location[trip_city_one_pre].y - location[trip_city_one_post].y) *
			(location[trip_city_one_pre].y - location[trip_city_one_post].y);
		//picking the first accepted and picking the last accepted is equivalent, and here I pick the latter one
		//because if I pick the small one, I have to tell whether the flag is 0
        if (proposal_dist < original_dist&&global_flag[0]<tid){
			global_flag[0] = tid;
			
		}
		__syncthreads();
		if (global_flag[0] == 0)
		{
			quotient = proposal_dist - original_dist;
			p = exp(-quotient / T[0]);
			myrandf = curand_uniform(&states[tid]);
			if (p > myrandf && global_flag[0]<tid){
				global_flag[0] = tid;
				__syncthreads();
			}
		}
		iter++;
		} 
	}
}

__global__ static void InsertionUpdateTrip(unsigned int* salesman_route, unsigned int* salesman_route2, unsigned int* __restrict__ N){

    unsigned int xid = blockIdx.x * blockDim.x + threadIdx.x;
    if (xid < N[0])
        salesman_route2[xid] = salesman_route[xid];
}

__global__ static void InsertionUpdate(unsigned int* __restrict__ city_one,
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

        if (city_one_swap < city_two_swap){
            if (xid >= city_one_swap && xid < city_two_swap){
                salesman_route[xid] = salesman_route2[xid + 1];
            }
			if (xid == 0)
				salesman_route[city_two_swap] = salesman_route2[city_one_swap];
        } else {
            if (xid > city_two_swap+1 && xid <= city_one_swap){
                salesman_route[xid] = salesman_route2[xid - 1];
            }
			if (xid == 0)
				salesman_route[city_two_swap + 1] = salesman_route2[city_one_swap];
        }
		__syncthreads();
		if (xid == 0)
			global_flag[0];
    }
}

__global__ static void tspInsertionUpdate(unsigned int* __restrict__ city_one,
                           unsigned int* __restrict__ city_two,
                           unsigned int* salesman_route,
                           volatile unsigned int *global_flag){
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tmp;
    int indicator;
    if (global_flag[0] != 0){
        indicator=city_one[global_flag[0]]-city_two[global_flag[0]];
        if(indicator>0)
        {
            if (tid<indicator)
            {
                if(tid==0)
                {
                    tmp = salesman_route[city_one[global_flag[0]]];
                }
                else

                {
                    tmp = salesman_route[city_two[global_flag[0]]+tid];
                }
                __syncthreads();
                salesman_route[tid+city_two[global_flag[0]]+1]=tmp;
            }
        }
        if(indicator<0)
        {
           if (tid<2-indicator)
           {
               if(tid==0) 
                {
                    tmp = salesman_route[city_one[global_flag[0]]];
                }
                else
                {
                    tmp = salesman_route[city_two[global_flag[0]]-tid+2];
                }
                __syncthreads();
                salesman_route[city_two[global_flag[0]]+1-tid]=tmp;
           }
        }
    }
    if(tid==0)
    {
        global_flag[0]=0;
    }
    __syncthreads();
}




#endif // _TSP_INSERT_H_

