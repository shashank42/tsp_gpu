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
- r: [unsigned integer(GRID_SIZE)]
- The random number to compare against for S.A.
*/
__global__ static void tsp(unsigned int* __restrict__ city_one,
	unsigned int* __restrict__ city_two,
	coordinates* __restrict__ location,
	unsigned int* __restrict__ salesman_route,
	float* __restrict__ T,
	float* __restrict__ r,
	unsigned int *flag,
	volatile unsigned int *global_flag){


	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float delta, p;
	unsigned int city_one_swap = city_one[tid];
	unsigned int city_two_swap = city_two[tid];

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



	if (proposal_dist < original_dist){
		flag[tid] = 1;
		global_flag[0] = 1;
	}
	else {
		delta = proposal_dist - original_dist;
		p = exp(-delta / T[0]);
		if (p > r[tid]){
			flag[tid] = 1;
			global_flag[0] = 1;
		}
	}
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
unsigned int rand_interval(unsigned int min, unsigned int max)
{
	int r;
	const unsigned int range = 1 + max - min;
	const unsigned int buckets = RAND_MAX / range;
	const unsigned int limit = buckets * range;

	/* Create equal size buckets all in a row, then fire randomly towards
	* the buckets until you land in one of them. All buckets are equally
	* likely. If you land off the end of the line of buckets, try again. */
	do
	{
		r = rand();
	} while (r >= limit);

	return min + (r / buckets);
}



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
	float T = 99999, T_start = 99999, *T_g, *r_g;
	float *r_h = (float *)malloc(GRID_SIZE * sizeof(float));
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
	unsigned int *city_swap_one_g, *city_swap_two_g, *salesman_route_g, *flag_g;
	unsigned int global_flag_h = 0, *global_flag_g;


	cudaError_t err = cudaMalloc((void**)&city_swap_one_g, GRID_SIZE * sizeof(unsigned int));
	//printf("\n Cuda malloc city swap one: %s \n", cudaGetErrorString(err));
	cudaMalloc((void**)&city_swap_two_g, GRID_SIZE * sizeof(unsigned int));
	cudaMalloc((void**)&location_g, N * sizeof(coordinates));
	cudaMalloc((void**)&salesman_route_g, N * sizeof(unsigned int));
	cudaMalloc((void**)&T_g, sizeof(float));
	cudaMalloc((void**)&r_g, GRID_SIZE * sizeof(float));
	cudaMalloc((void**)&flag_g, GRID_SIZE * sizeof(unsigned int));
	cudaMalloc((void**)&global_flag_g, sizeof(unsigned int));


	cudaMemcpy(location_g, location, N * sizeof(coordinates), cudaMemcpyHostToDevice);
	// Beta is the decay rate
	//float beta = 0.0001;
	// We are going to try some stuff for temp from this adaptive simulated annealing paper
	// https://arxiv.org/pdf/cs/0001018.pdf
	float new_loss_h = 0;
	float a = T_start;
	float f;
	float iter = 1.0;


	while (T > 90000){
		// Init parameters
		global_flag_h = 0;
		for (m = 0; m < GRID_SIZE; m++){
			// pick first city to swap
			city_swap_one_h[m] = rand_interval(1, N - 2);
			// f defines how far the second city can be from the first
			// This sort of gives us a nice slope
			//http://www.wolframalpha.com/input/?i=e%5E(-+sqrt(ln(9999%2Ft))))+from+9999+to+0
			f = exp(-sqrt(log(a / T)));
			j = (unsigned int)floor(1 + city_swap_one_h[m] * f);
			// pick second city to swap
			city_swap_two_h[m] = (city_swap_one_h[m] + j) % N;
			// Check we are not at the first or last city for city two
			if (city_swap_two_h[m] == 0)
				city_swap_two_h[m] += 1;
			if (city_swap_two_h[m] == N - 1)
				city_swap_two_h[m] -= 1;
			r_h[m] = (float)rand() / (float)RAND_MAX;

			//set our flags and new loss to 0
			flag_h[m] = 0;
		}
		// Copy memory from host to device
		err = cudaMemcpy(city_swap_one_g, city_swap_one_h, GRID_SIZE * sizeof(unsigned int), cudaMemcpyHostToDevice);
		//printf("\n Cuda mem copy city swap one: %s \n", cudaGetErrorString(err));
		cudaMemcpy(city_swap_two_g, city_swap_two_h, GRID_SIZE * sizeof(unsigned int), cudaMemcpyHostToDevice);
		cudaMemcpy(salesman_route_g, salesman_route, N * sizeof(unsigned int), cudaMemcpyHostToDevice);
		cudaMemcpy(T_g, &T, sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(r_g, r_h, GRID_SIZE * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(flag_g, flag_h, GRID_SIZE* sizeof(unsigned int), cudaMemcpyHostToDevice);
		cudaMemcpy(global_flag_g, &global_flag_h, sizeof(unsigned int), cudaMemcpyHostToDevice);

		// Number of thread blocks in grid
		dim3 blocksPerGrid(GRID_SIZE / t_num, 1, 1);
		dim3 threadsPerBlock(t_num, 1, 1);

		tsp << <blocksPerGrid, threadsPerBlock, 0 >> >(city_swap_one_g, city_swap_two_g,
			location_g, salesman_route_g,
			T_g, r_g, flag_g, global_flag_g);

		cudaThreadSynchronize();
		cudaMemcpy(&global_flag_h, global_flag_g, sizeof(unsigned int), cudaMemcpyDeviceToHost);
		if (global_flag_h != 0){
			cudaMemcpy(flag_h, flag_g, GRID_SIZE * sizeof(unsigned int), cudaMemcpyDeviceToHost);
			/*
			Here we check for a success
			The first proposal trip accepted becomes the new starting trip
			*/
			for (i = 0; i < GRID_SIZE; i++){
				if (flag_h[i] == 0){
					//printf("Original Loss: %.6f \n", original_loss);
					//printf("Proposed Loss: %.6f \n", new_loss_h[i]);
					continue;
				}
				else {
					// switch the two cities that led to an accepted proposal
					unsigned int tmp = salesman_route[city_swap_one_h[i]];
					salesman_route[city_swap_one_h[i]] = salesman_route[city_swap_two_h[i]];
					salesman_route[city_swap_two_h[i]] = tmp;
					new_loss_h = 0;
					for (i = 0; i < N - 1; i++){
						new_loss_h += (location[salesman_route[i]].x - location[salesman_route[i + 1]].x) *
							(location[salesman_route[i]].x - location[salesman_route[i + 1]].x) +
							(location[salesman_route[i]].y - location[salesman_route[i + 1]].y) *
							(location[salesman_route[i]].y - location[salesman_route[i + 1]].y);
					}

					// set old loss function to new
					original_loss = new_loss_h;
					//decrease temp
					T = T_start / iter;
					iter += 1.0f;
					if ((int)iter % 100 == 0){
						printf(" Current Temperature is %.6f \n", T);
						printf("\n Current Loss is: %.6f \n", original_loss);
						printf("\n Current Iteration is: %.6f \n", iter);
					}
					/*
					printf("Best found trip so far\n");
					for (j = 0; j < N; j++){
					printf("%d ", salesman_route[j]);
					}
					*/
					break;
				}
			}
		}
	}
	printf("The starting loss was %.6f and the final loss was %.6f \n", starting_loss, original_loss);
	/*
	printf("\n Final Route:\n");
	for (i = 0; i < N; i++)
	printf("%d ",salesman_route[i]);
	*/
	cudaFree(city_swap_one_g);
	cudaFree(city_swap_two_g);
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
	return 0;
}



