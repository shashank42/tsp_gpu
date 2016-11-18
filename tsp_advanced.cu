#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>


#define N 130 
#define t_num 1024
#define GRID_SIZE 2048

/*
For more samples define GRID_SIZE as a multiple of t_num such as 512000, 2048000, or the (max - 1024) grid size 2147482623
Some compliation options that can speed things up
--use_fast_math
--optimize=5
--gpu-architecture=compute_35
I use something like
nvcc --optimize=5 --use_fast_math -arch=compute_35 tsp_cuda.cu -o tsp_cuda
*/

// city's x y coordinates
typedef struct {
	float x;
	float y;
} coordinates;

// initialize the location by reading from file"ch130.tsp"
void initl(coordinates *c)
{
	FILE *fp;
	fp = fopen("ch130.tsp", "r");   //open file,read only
	int i;
	fscanf(fp, "%d", &i);
	while (i != 1)           //keep moving the pointer fp until it points to index 1
	{
		fseek(fp, 1, SEEK_CUR);
		fscanf(fp, "%d", &i);
	}
	fscanf(fp, "%f", &c[0].x);
	printf("%f\n", c[0].x);
	fscanf(fp, "%f", &c[0].y);
	printf("%f\n", c[0].y);
	while (i != N)
	{
		fscanf(fp, "%d", &i);   //index
		fscanf(fp, "%f", &c[i - 1].x);  //x
		fscanf(fp, "%f", &c[i - 1].y);  //y
	}

	fclose(fp);
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
*/
__global__ static void tsp(
	unsigned int* city_one,
	unsigned int* city_two,
	coordinates* __restrict__ location,
	unsigned int* __restrict__ salesman_route,
	float* __restrict__ T,
	int* __restrict__ seed,
	volatile unsigned int *global_flag){
	//first, refresh the routine, let thread 0 do it
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int tmp;
	if (tid == 0)
	{
		if (global_flag[0] != 0)
		{
			tmp = salesman_route[city_one[global_flag[0]]];
			salesman_route[city_one[global_flag[0]]] = salesman_route[city_two[global_flag[0]]];
			salesman_route[city_two[global_flag[0]]] = tmp;
			global_flag[0] = 0;
		}
	}
	__syncthreads();

	//second, we generate random number, get city_swap_index
	int a_r = 16807, q_r = 127773, z_r = 2836, r_r = seed[tid];   //note that these parameters' value can't be adjusted, they're optimum, by some mysterious math theory

	r_r = a_r*(r_r%q_r) - z_r*(r_r / q_r);
	if (r_r < 0)
		r_r += 2147483647;               //generate a random number

	int city_one_swap = (int)floor((float)r_r / 2147483647 * N);
	if (city_one_swap == 0)
		city_one_swap += 1;
	if (city_one_swap == N - 1)
		city_one_swap -= 1;

	r_r = a_r*(r_r%q_r) - z_r*(r_r / q_r);
	if (r_r < 0)
		r_r += 2147483647;                         //generate a new random number

	int city_two_swap = ((int)(city_one_swap + (int)floor(((float)r_r / 2147483647 * 2 - 1)*N*exp(-1 / T[0]))) + N) % N;
	if (city_two_swap == 0)
		city_two_swap += 1;
	if (city_two_swap == N - 1)
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
	}
	else {
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



/* Function to generate random numbers in interval

input:
- min [unsigned integer(1)]
- The minimum number to sample
- max [unsigned integer(1)]
- The maximum number to sample

Output: [unsigned integer(1)]
- A randomly generated number between the range of min and max

Desc:
Taken from
- http://stackoverflow.com/questions/2509679/how-to-generate-a-random-number-from-within-a-range


*/
/*  It's no longer needed
unsigned int rand_interval(unsigned int min, unsigned int max)
{
	int r;
	const unsigned int range = 1 + max - min;
	const unsigned int buckets = RAND_MAX / range;
	const unsigned int limit = buckets * range;

	/* Create equal size buckets all in a row, then fire randomly towards
	* the buckets until you land in one of them. All buckets are equally
	* likely. If you land off the end of the line of buckets, try again. */
/*
	do
	{
		r = rand();
	} while (r >= limit);

	return min + (r / buckets);
}
*/


int main(){

	// start counters for cities
	unsigned int i, j, m;

	coordinates *location, *location_g;
	location = (coordinates *)malloc(N * sizeof(coordinates));

	unsigned int *salesman_route = (unsigned int *)malloc(N * sizeof(unsigned int));

	// just make one inital guess route, a simple linear path
	for (i = 0; i < N; i++)
		salesman_route[i] = i;

	// Set the starting and end points to be the same
	salesman_route[N - 1] = salesman_route[0];

	/*     don't need it when importing data from files
	// initialize the coordinates and sequence
	for(i = 0; i < N; i++){
	location[i].x = rand() % 1000;
	location[i].y = rand() % 1000;
	}
	*/

	initl(location);

	// Calculate the original loss
	float original_loss = 0;
	for (i = 0; i < N - 1; i++){
		original_loss += (location[salesman_route[i]].x - location[salesman_route[i + 1]].x) *
			(location[salesman_route[i]].x - location[salesman_route[i + 1]].x) +
			(location[salesman_route[i]].y - location[salesman_route[i + 1]].y) *
			(location[salesman_route[i]].y - location[salesman_route[i + 1]].y);
	}
	printf("Original Loss is: %.6f \n", original_loss);
	// Keep the original loss for comparison pre/post algorithm
	float starting_loss = original_loss;
	float T = 99999, T_start = 99999, *T_g;
	int *r_g;
	int *r_h = (int *)malloc(GRID_SIZE * sizeof(int));
	for (i = 0; i<GRID_SIZE; i++)
	{
		r_h[i] = rand();
	}
	/*
	Defining device variables:
	city_swap_one_h/g: [integer(t_num)]
	- Host/Device memory for city one
	city_swap_two_h/g: [integer(t_num)]
	- Host/Device memory for city two
	flag_h/g: [integer(t_num)]
	- Host/Device memory for flag of accepted step
	salesman_route_g: [integer(N)]
	- Device memory for the salesmans route
	r_g:  [float(t_num)]
	- Device memory for the random number when deciding acceptance
	flag_h/g: [integer(t_num)]
	- host/device memory for acceptance vector
	original_loss_g: [integer(1)]
	- The device memory for the current loss function
	new_loss_h/g: [integer(t_num)]
	- The host/device memory for the proposal loss function
	*/
	unsigned int *city_swap_one_h = (unsigned int *)malloc(GRID_SIZE * sizeof(unsigned int));
	unsigned int *city_swap_two_h = (unsigned int *)malloc(GRID_SIZE * sizeof(unsigned int));
	unsigned int *flag_h = (unsigned int *)malloc(GRID_SIZE * sizeof(unsigned int));
	unsigned int *salesman_route_g, *flag_g, *city_swap_one_g, *city_swap_two_g;
	unsigned int global_flag_h = 0, *global_flag_g;

	cudaMalloc((void**)&city_swap_one_g, GRID_SIZE * sizeof(unsigned int));
	cudaMalloc((void**)&city_swap_two_g, GRID_SIZE * sizeof(unsigned int));
	cudaMalloc((void**)&salesman_route_g, N * sizeof(unsigned int));
	cudaMalloc((void**)&location_g, N * sizeof(coordinates));
	cudaMalloc((void**)&salesman_route_g, N * sizeof(unsigned int));
	cudaMalloc((void**)&T_g, sizeof(float));
	cudaMalloc((void**)&r_g, GRID_SIZE * sizeof(int));
	cudaMalloc((void**)&flag_g, GRID_SIZE * sizeof(unsigned int));
	cudaMalloc((void**)&global_flag_g, sizeof(unsigned int));


	cudaMemcpy(location_g, location, N * sizeof(coordinates), cudaMemcpyHostToDevice);
	cudaMemcpy(salesman_route_g, salesman_route, N * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(r_g, r_h, GRID_SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(global_flag_g, &global_flag_h, sizeof(unsigned int), cudaMemcpyHostToDevice);
	// Beta is the decay rate
	//float beta = 0.0001;
	// We are going to try some stuff for temp from this adaptive simulated annealing paper
	// https://arxiv.org/pdf/cs/0001018.pdf
	float new_loss_h = 0;
	float a = T_start;
	float f;
	float iter = 1.0;

	// Number of thread blocks in grid
	dim3 blocksPerGrid(GRID_SIZE / t_num, 1, 1);
	dim3 threadsPerBlock(t_num, 1, 1);

	while (T > 1000){
		// Init parameters
		global_flag_h = 0;
		// Copy memory from host to device
		cudaMemcpy(T_g, &T, sizeof(float), cudaMemcpyHostToDevice);

		tsp << <blocksPerGrid, threadsPerBlock, 0 >> >(city_swap_one_g, city_swap_two_g, location_g, salesman_route_g,
			T_g, r_g, global_flag_g);

		cudaThreadSynchronize();
	//		cudaMemcpy(&global_flag_h, global_flag_g, sizeof(unsigned int), cudaMemcpyDeviceToHost);
		T = T*0.9999;
		//	printf("%d\n",global_flag_h);
	}

	cudaMemcpy(salesman_route, salesman_route_g, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	float optimized_loss = 0;
	for (i = 0; i < N - 1; i++){
		optimized_loss += (location[salesman_route[i]].x - location[salesman_route[i + 1]].x) *
			(location[salesman_route[i]].x - location[salesman_route[i + 1]].x) +
			(location[salesman_route[i]].y - location[salesman_route[i + 1]].y) *
			(location[salesman_route[i]].y - location[salesman_route[i + 1]].y);
	}
		printf("Optimized Loss is: %.6f \n", optimized_loss);
	
	/*
	printf("\n Final Route:\n");
	for (i = 0; i < N; i++)
	printf("%d ",salesman_route[i]);
	*/
	cudaFree(location_g);
	cudaFree(salesman_route_g);
	cudaFree(T_g);
	cudaFree(r_g);
	cudaFree(flag_g);
	free(salesman_route);
	free(city_swap_one_h);
	free(city_swap_two_h);
	free(flag_h);
	free(location);
	getchar();
	return 0;
}

