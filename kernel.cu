#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>
#include <string.h>
#include <limits.h>
#include <assert.h>
#include <iostream>
#include <fstream>


#include "utils.h"
#include "tsp_solve.h"


#define t_num 1024
#define GRID_SIZE 9216

/*
For more samples define GRID_SIZE as a multiple of t_num such as 512000, 2048000, or the (max - 1024) grid size 2147482623
Some compliation options that can speed things up
--use_fast_math
--optimize=5
--gpu-architecture=compute_35
I use something like
NOTE: You need to use the -lcurand flag to compile.
nvcc --optimize=5 --use_fast_math -arch=compute_35 kernel.cu -o tsp_cuda -lcurand
*/

int main(){

	const char *tsp_name = "dsj1000.tsp";
	read_tsp(tsp_name);
	unsigned int N = meta->dim, *N_g;
	// start counters for cities
	unsigned int i;

	coordinates *location_g;

	/* For checking the coordinates
	for (i = 0; i < N; i++)
	printf("Location x: %0.6f, location y: %0.6f \n", location[i].x, location[i].y);
	exit(0);
	*/
	unsigned int *salesman_route = (unsigned int *)malloc((N + 1) * sizeof(unsigned int));

	// just make one inital guess route, a simple linear path
	for (i = 0; i <= N; i++)
		salesman_route[i] = i;

	// Set the starting and end points to be the same
	salesman_route[N] = salesman_route[0];

	/*     don't need it when importing data from files
	// initialize the coordinates and sequence
	for(i = 0; i < N; i++){
	location[i].x = rand() % 1000;
	location[i].y = rand() % 1000;
	}
	*/



	// Calculate the original loss
	float original_loss = 0;
	for (i = 0; i < N; i++){
		original_loss += (location[salesman_route[i]].x - location[salesman_route[i + 1]].x) *
			(location[salesman_route[i]].x - location[salesman_route[i + 1]].x) +
			(location[salesman_route[i]].y - location[salesman_route[i + 1]].y) *
			(location[salesman_route[i]].y - location[salesman_route[i + 1]].y);
	}
	printf("Original Loss is:  %0.6f \n", original_loss);
	// Keep the original loss for comparison pre/post algorithm
	// SET THE LOSS HERE
	float T[1], *T_g;
	T[0] = 30;
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
	flag_h/g: [integer(t_num)]
	- host/device memory for acceptance vector
	original_loss_g: [integer(1)]
	- The device memory for the current loss function
	(DEPRECATED)new_loss_h/g: [integer(t_num)]
	- The host/device memory for the proposal loss function
	*/
	unsigned int *city_swap_one_h = (unsigned int *)malloc(GRID_SIZE * sizeof(unsigned int));
	unsigned int *city_swap_two_h = (unsigned int *)malloc(GRID_SIZE * sizeof(unsigned int));
	unsigned int *flag_h = (unsigned int *)malloc(GRID_SIZE * sizeof(unsigned int));
	unsigned int *salesman_route_g, *salesman_route_2g, *flag_g, *city_swap_one_g, *city_swap_two_g;
	unsigned int global_flag_h = 0, *global_flag_g;

	cudaMalloc((void**)&city_swap_one_g, GRID_SIZE * sizeof(unsigned int));
	cudaCheckError();
	cudaMalloc((void**)&city_swap_two_g, GRID_SIZE * sizeof(unsigned int));
	cudaCheckError();
	cudaMalloc((void**)&location_g, N * sizeof(coordinates));
	cudaCheckError();
	cudaMalloc((void**)&salesman_route_g, (N + 1) * sizeof(unsigned int));
	cudaCheckError();
	cudaMalloc((void**)&salesman_route_2g, (N + 1) * sizeof(unsigned int));
	cudaCheckError();
	cudaMalloc((void**)&T_g, sizeof(float));
	cudaCheckError();
	cudaMalloc((void**)&flag_g, GRID_SIZE * sizeof(unsigned int));
	cudaCheckError();
	cudaMalloc((void**)&global_flag_g, sizeof(unsigned int));
	cudaCheckError();
	cudaMalloc((void**)&N_g, sizeof(unsigned int));
	cudaCheckError();


	cudaMemcpy(location_g, location, N * sizeof(coordinates), cudaMemcpyHostToDevice);
	cudaCheckError();
	cudaMemcpy(salesman_route_g, salesman_route, (N + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaCheckError();
	cudaMemcpy(salesman_route_2g, salesman_route, (N + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaCheckError();
	cudaMemcpy(global_flag_g, &global_flag_h, sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaCheckError();
	cudaMemcpy(N_g, &N, sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaCheckError();
	// Beta is the decay rate
	//float beta = 0.0001;
	// We are going to try some stuff for temp from this adaptive simulated annealing paper
	// https://arxiv.org/pdf/cs/0001018.pdf

	// Number of thread blocks in grid
	// X is for the sampling, y is for manipulating the salesman's route
	dim3 blocksPerSampleGrid(GRID_SIZE / t_num, 1, 1);
	dim3 blocksPerTripGrid((N / t_num) + 1, 1, 1);
	dim3 threadsPerBlock(t_num, 1, 1);

	// Trying out random gen in cuda
	curandState_t* states;

	/* allocate space on the GPU for the random states */
	cudaMalloc((void**)&states, GRID_SIZE * sizeof(curandState_t));
	init <<<blocksPerSampleGrid, threadsPerBlock, 0 >>>(time(0), states);

	//time counter
	time_t t_start, t_end;
	t_start = time(NULL);

	while (T[0] > 1)
	{
		// Copy memory from host to device
		cudaMemcpy(T_g, T, sizeof(float), cudaMemcpyHostToDevice);
		cudaCheckError();
		i = 1;

		while (i<500){

			
			tspSwap <<<blocksPerSampleGrid, threadsPerBlock, 0 >>>(city_swap_one_g, city_swap_two_g,
				                                                   location_g, salesman_route_g,
				                                                   T_g, global_flag_g, N_g,
				                                                   states);
			cudaCheckError(); 
			tspSwapUpdate <<<blocksPerSampleGrid, threadsPerBlock, 0 >>>(city_swap_one_g, city_swap_two_g,
				                                                         salesman_route_g, global_flag_g);
			cudaCheckError();
			
			tspSwap2 << <blocksPerSampleGrid, threadsPerBlock, 0 >> >(city_swap_one_g, city_swap_two_g,
				                                                      location_g, salesman_route_g,
				                                                      T_g, global_flag_g, N_g,
				                                                      states);
			cudaCheckError();
			
			tspSwapUpdate << <blocksPerSampleGrid, threadsPerBlock, 0 >> >(city_swap_one_g, city_swap_two_g,
				salesman_route_g, global_flag_g);
			cudaCheckError();
			/*

			tspInsertion <<<blocksPerSampleGrid, threadsPerBlock, 0 >>>(city_swap_one_g, city_swap_two_g,
			                                                            location_g, salesman_route_g,
			                                                            T_g, global_flag_g, N_g,
			                                                            states); 
			cudaCheckError();
			
			tspInsertionUpdateTrip << <blocksPerTripGrid, threadsPerBlock, 0 >> >(salesman_route_g, salesman_route_2g, N_g);
			cudaCheckError();
			
			tspInsertionUpdate2 <<<blocksPerTripGrid, threadsPerBlock, 0 >>>(city_swap_one_g, city_swap_two_g,
				                                                             salesman_route_g, salesman_route_2g,
				                                                             global_flag_g);
            cudaCheckError();
            
		    tspInsertionUpdateTrip << <blocksPerTripGrid, threadsPerBlock, 0 >> >(salesman_route_g, salesman_route_2g, N_g);
			cudaCheckError();

			tspInsertion2 << <blocksPerSampleGrid, threadsPerBlock, 0 >> >(city_swap_one_g, city_swap_two_g,
				                                                           location_g, salesman_route_g,
				                                                           T_g, global_flag_g, N_g,
				                                                           states);
			cudaCheckError();

			tspInsertionUpdate2 << <blocksPerTripGrid, threadsPerBlock, 0 >> >(city_swap_one_g, city_swap_two_g,
				                                                               salesman_route_g, salesman_route_2g,
				                                                               global_flag_g);
			cudaCheckError();
			
			tspInsertionUpdateTrip << <blocksPerTripGrid, threadsPerBlock, 0 >> >(salesman_route_g, salesman_route_2g, N_g);
			cudaCheckError();
			*/
	//		iter += 1.00f;
	//		T = T_start / log(iter);
	//		if ((long int)iter % 50000 == 0)
	//			printf("Iter: %ld  Temperature is %.6f\n", (long int)iter, T);
			//T = 1;
			i++;
		}
		cudaMemcpy(salesman_route, salesman_route_g, (N + 1) * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		cudaCheckError();
		float optimized_loss = 0;
		for (i = 0; i < N; i++){
			optimized_loss += (location[salesman_route[i]].x - location[salesman_route[i + 1]].x) *
				(location[salesman_route[i]].x - location[salesman_route[i + 1]].x) +
				(location[salesman_route[i]].y - location[salesman_route[i + 1]].y) *
				(location[salesman_route[i]].y - location[salesman_route[i + 1]].y);
		}
		
		T[0] = T[0] * 0.999;
		printf("T[0]: %f Optimized Loss is: %.6f  \n",T[0], optimized_loss);
	}
	//print time spent
	t_end = time(NULL);
	printf("time = %f\n", difftime(t_end, t_start));

	cudaMemcpy(salesman_route, salesman_route_g, (N + 1) * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaCheckError();
	float optimized_loss = 0;
	for (i = 0; i < N; i++){
		optimized_loss += (location[salesman_route[i]].x - location[salesman_route[i + 1]].x) *
			(location[salesman_route[i]].x - location[salesman_route[i + 1]].x) +
			(location[salesman_route[i]].y - location[salesman_route[i + 1]].y) *
			(location[salesman_route[i]].y - location[salesman_route[i + 1]].y);
	} 
	printf("Optimized Loss is: %.6f \n", optimized_loss);

	// Write the best trip to CSV
	FILE *best_trip;
	const char *filename = "mona_lisa_best_trip.csv";
	best_trip = fopen(filename, "w+");
	fprintf(best_trip, "location,coordinate_x,coordinate_y\n");
	for (i = 0; i < N + 1; i++){
		fprintf(best_trip, "%d,%.6f,%.6f\n",
			salesman_route[i],
			location[salesman_route[i]].x,
			location[salesman_route[i]].y);
	}
	fclose(best_trip);

	/*
	printf("\n Final Route:\n");
	for (i = 0; i < N; i++)
	printf("%d ",salesman_route[i]);
	*/
	cudaFree(location_g);
	cudaCheckError();
	cudaFree(salesman_route_g);
	cudaCheckError();
	cudaFree(salesman_route_2g);
	cudaCheckError();
	cudaFree(T_g);
	cudaCheckError();
	cudaFree(flag_g);
	cudaCheckError();
	free(salesman_route);
	free(city_swap_one_h);
	free(city_swap_two_h);
	free(flag_h);
	free(location);
	getchar();
	return 0;
}

