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


__global__ static void globalSwap_f(unsigned int* city_one,
                           unsigned int* city_two,
                           coordinates* __restrict__ location,
                           unsigned int* __restrict__ salesman_route,
                           float* __restrict__ T,
                           volatile unsigned int *global_flag,
                           unsigned int* __restrict__ N,
                           curandState_t* states){

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int iter = 0;
    // This is the maximum we can sample from
    // This gives us a nice curve
    //http://www.wolframalpha.com/input/?i=e%5E(-+10%2Ft)+from+10+to+1
    int sample_space = (int)floor(exp(- (T[1] / 15) / T[0]) * (float)N[0] + 3);
    // Run until either global flag is zero and we do 100 iterations is false.
    while (global_flag[0] == 0 && iter < 3){

    float myrandf = curand_uniform(&states[tid]);
    myrandf *= ((float)(N[0] - 4) - 1.0+0.9999999999999999);
    myrandf += 1.0;
    int city_one_swap = (int)truncf(myrandf);

    //because we need only positive part, normal dist is not ideal.
    // space have to be lesser than N-2, there will be error when
    // city_one=1 and city_two=N-1, that's also an edge condition
    // anyway that's not a problem because I saw your sample space
    // parameter starts from a small percentage, that's fine
	int min_city_two = city_one_swap + 3;
	//this is the key change, we fix city_two larger than city_one
	//so we won;t sample cities that near city one!
    int max_city_two = (city_one_swap +3+ sample_space < N[0])?
        city_one_swap +3+ sample_space:
            (N[0] - 1);
    myrandf = curand_uniform(&states[tid]);
    myrandf *= ((float)max_city_two - (float)min_city_two + 0.999999999999999);
    myrandf += min_city_two;
    int city_two_swap = (int)truncf(myrandf);

    //by this sampling method there's strictly no increase
    //because all loss change calculation is correct now!



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
        p = exp(-(quotient * T[1] * 300) / T[0]);
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

__global__ static void globalSwap_b(unsigned int* city_one,
                           unsigned int* city_two,
                           coordinates* __restrict__ location,
                           unsigned int* __restrict__ salesman_route,
                           float* __restrict__ T,
                           volatile unsigned int *global_flag,
                           unsigned int* __restrict__ N,
                           curandState_t* states){

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int iter = 0;
    // This is the maximum we can sample from
    // This gives us a nice curve
    //http://www.wolframalpha.com/input/?i=e%5E(-+10%2Ft)+from+10+to+1
    int sample_space = (int)floor(exp(- (T[1] / 15) / T[0]) * (float)N[0] + 3);
    // Run until either global flag is zero and we do 100 iterations is false.
    while (global_flag[0] == 0 && iter < 3){

    float myrandf = curand_uniform(&states[tid]);
    myrandf *= ((float)(N[0] - 1) - 4.0+0.9999999999999999);
    myrandf += 4.0;
    int city_one_swap = (int)truncf(myrandf);

    //just reverse the min and max's expression
	int max_city_two = city_one_swap - 3;

    int min_city_two = (city_one_swap -3- sample_space > 0)?
        city_one_swap -3- sample_space:
            1;
    myrandf = curand_uniform(&states[tid]);
    myrandf *= ((float)max_city_two - (float)min_city_two + 0.999999999999999);
    myrandf += min_city_two;
    int city_two_swap = (int)truncf(myrandf);

    //by this sampling method there's strictly no increase
    //because all loss change calculation is correct now!



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
        p = exp(-(quotient * T[1] * 300) / T[0]);
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

__global__ static void localSwap_f(unsigned int* city_one,
	unsigned int* city_two,
	coordinates* __restrict__ location,
	unsigned int* __restrict__ salesman_route,
	float* __restrict__ T,
	volatile unsigned int *global_flag,
	unsigned int* __restrict__ N,
	curandState_t* states){

	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int iter = 0;
    // This is the maximum we can sample from
    // This gives us a nice curve
    //http://www.wolframalpha.com/input/?i=e%5E(-+10%2Ft)+from+10+to+1
    int sample_space = (int)floor(exp(- (T[1]/2) / T[0]) * (float)N[0] + 3);
    // Run until either global flag is zero and we do 100 iterations is false.
    while (global_flag[0] == 0 && iter < 3){



    float myrandf = curand_uniform(&states[tid]);
    myrandf *= ((float)(N[0] - 4) - 1.0+0.9999999999999999);
    myrandf += 1.0;
    int city_one_swap = (int)truncf(myrandf);

    //because we need only positive part, normal dist is not ideal.
    // space have to be lesser than N-2, there will be error when
    // city_one=1 and city_two=N-1, that's also an edge condition
    // anyway that's not a problem because I saw your sample space
    // parameter starts from a small percentage, that's fine
	int min_city_two = city_one_swap + 3;
	//this is the key change, we fix city_two larger than city_one
	//so we won;t sample cities that near city one!
    int max_city_two = (city_one_swap +3+ sample_space < N[0])?
        city_one_swap +3+ sample_space:
            (N[0] - 1);
    myrandf = curand_uniform(&states[tid]);
    myrandf *= ((float)max_city_two - (float)min_city_two + 0.999999999999999);
    myrandf += min_city_two;
    int city_two_swap = (int)truncf(myrandf);

    //by this sampling method there's strictly no increase
    //because all loss change calculation is correct now!


        city_one[tid] = city_one_swap;
        city_two[tid] = city_two_swap;

		float quotient, p;
		// Set the swap cities in the trip.
		unsigned int trip_city_one = salesman_route[city_one_swap];
		unsigned int trip_city_one_pre = salesman_route[city_one_swap - 1];
		unsigned int trip_city_one_post = salesman_route[city_one_swap + 1];
		unsigned int trip_city_two = salesman_route[city_two_swap];
		unsigned int trip_city_two_pre = salesman_route[city_two_swap - 1];
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
			__syncthreads();
		}
		if (T[0] > 1){
		if (global_flag[0] == 0){
			quotient = proposal_dist / original_dist - 1;
            // You can change the constant to whatever you would like
			// But you should check that the graph looks nice
			//http://www.wolframalpha.com/input/?i=e%5E(-(x*(10000%2F5))%2Ft)+x+%3D+0+to+3+and+t+%3D+0+to+10000
            p = exp(-(quotient * T[1] * 300) / T[0]);
            myrandf = curand_uniform(&states[tid]);
			if (p > myrandf && global_flag[0]<tid){
				global_flag[0] = tid;
				__threadfence();
			}
		}
		}
		iter++;
	}
}

__global__ static void localSwap_b(unsigned int* city_one,
	unsigned int* city_two,
	coordinates* __restrict__ location,
	unsigned int* __restrict__ salesman_route,
	float* __restrict__ T,
	volatile unsigned int *global_flag,
	unsigned int* __restrict__ N,
	curandState_t* states){

	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int iter = 0;
    // This is the maximum we can sample from
    // This gives us a nice curve
    //http://www.wolframalpha.com/input/?i=e%5E(-+10%2Ft)+from+10+to+1
    int sample_space = (int)floor(exp(- (T[1]/2) / T[0]) * (float)N[0] + 3);
    // Run until either global flag is zero and we do 100 iterations is false.
    while (global_flag[0] == 0 && iter < 3){



    float myrandf = curand_uniform(&states[tid]);
    myrandf *= ((float)(N[0] - 1) - 4.0+0.9999999999999999);
    myrandf += 4.0;
    int city_one_swap = (int)truncf(myrandf);

    //just reverse the min and max's expression
	int max_city_two = city_one_swap - 3;

    int min_city_two = (city_one_swap -3- sample_space > 0)?
        city_one_swap - 3 - sample_space:
            1;
    myrandf = curand_uniform(&states[tid]);
    myrandf *= ((float)max_city_two - (float)min_city_two + 0.999999999999999);
    myrandf += min_city_two;
    int city_two_swap = (int)truncf(myrandf);

    //by this sampling method there's strictly no increase
    //because all loss change calculation is correct now!



        city_one[tid] = city_one_swap;
        city_two[tid] = city_two_swap;

		float quotient, p;
		// Set the swap cities in the trip.
		unsigned int trip_city_one = salesman_route[city_one_swap];
		unsigned int trip_city_one_pre = salesman_route[city_one_swap - 1];
		unsigned int trip_city_one_post = salesman_route[city_one_swap + 1];
		unsigned int trip_city_two = salesman_route[city_two_swap];
		unsigned int trip_city_two_pre = salesman_route[city_two_swap - 1];
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
			__syncthreads();
		}
		if (T[0] > 1){
		if (global_flag[0] == 0){
			quotient = proposal_dist / original_dist - 1;
            // You can change the constant to whatever you would like
			// But you should check that the graph looks nice
			//http://www.wolframalpha.com/input/?i=e%5E(-(x*(10000%2F5))%2Ft)+x+%3D+0+to+3+and+t+%3D+0+to+10000
            p = exp(-(quotient * T[1] * 300) / T[0]);
            myrandf = curand_uniform(&states[tid]);
			if (p > myrandf && global_flag[0]<tid){
				global_flag[0] = tid;
				__threadfence();
			}
		}
		}
		iter++;
	}
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

