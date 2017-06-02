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

__global__ static void insertionStep(unsigned int* city_one,
	unsigned int* city_two,
	coordinates* __restrict__ location,
	unsigned int* __restrict__ salesman_route,
	float* __restrict__ T,
	volatile int *global_flag,
	unsigned int* __restrict__ N,
	curandState_t* states,
	float* sample_area){
	//first, refresh the route, this time we have to change city_one-city_two elements
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	//if (tid == 0)
	//	global_flag[0] = -1;

    // This is the maximum we can sample from
	int sample_space = (int)floor(5 + exp(- sample_area[0] / T[0]) * N[0]);
	// Generate the first city
	// From: http://stackoverflow.com/questions/18501081/generating-random-number-within-cuda-kernel-in-a-varying-range
	float myrandf = curand_uniform(&states[tid]);
	myrandf *= ((float)(N[0] - 2) - 4.0 + 0.9999999999999999);
	myrandf += 4.0;
	int city_one_swap = (int)truncf(myrandf);
    
    city_one_swap = (city_one_swap > N[0] - 2) * (N[0] - 2) + !(city_one_swap > N[0] - 2) * city_one_swap;
    city_one_swap = (city_one_swap < 4) * 4 + !(city_one_swap < 4) * city_one_swap;    
	// Trying out normally distributed swap step
    int city_two_swap = (int)(city_one_swap + (curand_normal(&states[tid]) * sample_space));
    
    // One is added here so that if we have city two == N, then it bumps it up to 1
    city_two_swap -= (city_two_swap >= N[0]) * ((city_two_swap/N[0]) * N[0] + 1);
    // this is N[0] - 2 because we cannot have a value of N[0] - 1,
    // So 0 + N[0] -> N[0] - 1, normally, but doing -2 gives N[0] - 2
    city_two_swap += (city_two_swap <= 0) * ((-city_two_swap/N[0] + 1) * N[0] - 2);
    
       
    // Check it ||city_two - city one|| < 9, if so bump it up one
    //city_two_swap = ((city_one_swap - city_two_swap) * (city_one_swap - city_two_swap) < 9) * (city_one_swap - 3) + !((city_one_swap - city_two_swap) * (city_one_swap - city_two_swap) < 9) * city_two_swap;
    
    city_two_swap = (city_two_swap > N[0] - 2) * (N[0] - 2) + !(city_two_swap > N[0] - 2) * city_two_swap;
    city_two_swap = (city_two_swap < 1) * 1 + !(city_two_swap < 1) * city_two_swap;

/*
	// We need to set the min and max of the second city swap
	int max_city_two = city_one_swap - 3;

	int min_city_two = (city_one_swap - 3 - sample_space >= 0) ?
		city_one_swap - 3 - sample_space :
		0;
	myrandf = curand_uniform(&states[tid]);
	myrandf *= ((float)max_city_two - (float)min_city_two + 0.999999999999999);
	myrandf += min_city_two;
	int city_two_swap = (int)truncf(myrandf);

*/

	if (city_two_swap != (N[0] - 1) && city_two_swap != city_one_swap && city_two_swap != city_one_swap - 1)
	{
		city_one[tid] = city_one_swap;
		city_two[tid] = city_two_swap;

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
		original_dist += sqrtf((location[trip_city_one_pre].x - location[trip_city_one].x) *
			(location[trip_city_one_pre].x - location[trip_city_one].x) +
			(location[trip_city_one_pre].y - location[trip_city_one].y) *
			(location[trip_city_one_pre].y - location[trip_city_one].y));
		original_dist += sqrtf((location[trip_city_one_post].x - location[trip_city_one].x) *
			(location[trip_city_one_post].x - location[trip_city_one].x) +
			(location[trip_city_one_post].y - location[trip_city_one].y) *
			(location[trip_city_one_post].y - location[trip_city_one].y));
		original_dist += sqrtf((location[trip_city_two_post].x - location[trip_city_two].x) *
			(location[trip_city_two_post].x - location[trip_city_two].x) +
			(location[trip_city_two_post].y - location[trip_city_two].y) *
			(location[trip_city_two_post].y - location[trip_city_two].y));

		proposal_dist += sqrtf((location[trip_city_two].x - location[trip_city_one].x) *
			(location[trip_city_two].x - location[trip_city_one].x) +
			(location[trip_city_two].y - location[trip_city_one].y) *
			(location[trip_city_two].y - location[trip_city_one].y));
		proposal_dist += sqrtf((location[trip_city_two_post].x - location[trip_city_one].x) *
			(location[trip_city_two_post].x - location[trip_city_one].x) +
			(location[trip_city_two_post].y - location[trip_city_one].y) *
			(location[trip_city_two_post].y - location[trip_city_one].y));
		proposal_dist += sqrtf((location[trip_city_one_pre].x - location[trip_city_one_post].x) *
			(location[trip_city_one_pre].x - location[trip_city_one_post].x) +
			(location[trip_city_one_pre].y - location[trip_city_one_post].y) *
			(location[trip_city_one_pre].y - location[trip_city_one_post].y));
		//picking the first accepted and picking the last accepted is equivalent, and here I pick the latter one
		//because if I pick the small one, I have to tell whether the flag is 0
		if (proposal_dist < original_dist&&global_flag[0] == -1)
		{
			global_flag[0] = tid;
		}
		else if (global_flag[0] == -1)
		{
		    double quotient, p;
		    quotient = proposal_dist - original_dist;
		    p = exp(-(quotient) / T[0]);
		    myrandf = curand_uniform(&states[tid]);
		    if (p/131072 > myrandf && global_flag[0] == -1)
		    {
		        global_flag[0] = tid;
		    }
		}
	}
}


__global__ static void insertionUpdateTrip(unsigned int* salesman_route, unsigned int* salesman_route2, unsigned int* __restrict__ N){

    unsigned int xid = blockIdx.x * blockDim.x + threadIdx.x;
    if (xid < N[0])
        salesman_route2[xid] = salesman_route[xid];
}

__global__ static void insertionUpdate(unsigned int* __restrict__ city_one,
                           unsigned int* __restrict__ city_two,
                           unsigned int* salesman_route,
                           unsigned int* salesman_route2,
                           volatile int* global_flag){

    // each thread is a position in the salesman's trip
    const int xid = blockIdx.x * blockDim.x + threadIdx.x;
    /*
      1. Save city one
      2. Shift everything between city one and city two up or down, depending on city one < city two
      3. Set city two's old position to city one
    */
    if (global_flag[0] != -1){
        unsigned int city_one_swap = city_one[global_flag[0]];
        unsigned int city_two_swap = city_two[global_flag[0]];

        if (city_one_swap < city_two_swap){
            if (xid >= city_one_swap && xid < city_two_swap){
                salesman_route[xid] = salesman_route2[xid + 1];
            }
        } else {
            if (xid > city_two_swap+1 && xid <= city_one_swap){
                salesman_route[xid] = salesman_route2[xid - 1];
            }
        }
    }
}

__global__ static void insertionUpdateEndPoints(unsigned int* __restrict__ city_one,
                           unsigned int* __restrict__ city_two,
                           unsigned int* salesman_route,
                           unsigned int* salesman_route2,
                           volatile int* global_flag){
    // each thread is a position in the salesman's trip
    const int xid = blockIdx.x * blockDim.x + threadIdx.x;
    /*
      1. Save city one
      2. Shift everything between city one and city two up or down, depending on city one < city two
      3. Set city two's old position to city one
    */
    if (global_flag[0] != -1){
        unsigned int city_one_swap = city_one[global_flag[0]];
        unsigned int city_two_swap = city_two[global_flag[0]];

        if (city_one_swap < city_two_swap){
            if (xid == 0)
				salesman_route[city_two_swap] = salesman_route2[city_one_swap];
        } else {
            if (xid == 0)
				salesman_route[city_two_swap + 1] = salesman_route2[city_one_swap];
        }
    }
}


/*
__global__ static void insertionUpdate(unsigned int* __restrict__ city_one,
                           unsigned int* __restrict__ city_two,
                           unsigned int* salesman_route,
                           volatile int* global_flag){
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tmp;
    int indicator;
    if (global_flag[0] != -1){
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
        global_flag[0]=-1;
    }
    __syncthreads();
}


*/

#endif // _TSP_INSERT_H_

