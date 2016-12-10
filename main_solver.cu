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
#include <ctype.h>
#include "kernels/utils.h"
#include "kernels/initialize_rng.h"
#include "kernels/swap_sampler.h"
#include "kernels/insert_sampler.h"
#include "kernels/opt2_sampler.h"


#define t_num 1024
#define GRID_SIZE 131072

/*
For more samples define GRID_SIZE as a multiple of t_num such as 512000, 2048000, or the (max - 1024) grid size 2147482623
A good grid size is the number of SM's you have times the number of blocks each can take in times max threads per block
I have 8 cores that can hold 16 blocks of 1024 cores so my best is 131072
Some compiler options that can speed things up
--use_fast_math
--optimize=5
--gpu-architecture=compute_35
I use something like
NOTE: You need to use the -lcurand flag to compile for the RNG.
nvcc --optimize=5 --use_fast_math -arch=compute_35 main_solver.cu -o tsp_cuda -lcurand
*/

int main(int argc, char *argv[]){

// Reading in inputs
    if (argc == 1){
        printf("Inputs: \n" 
               "(Required) input_file.tsp: [char()] \n"
               " - The name of the tsp file, excluding .tsp at the end, containing the cities to travel over. \n"
               "(Optional) -trip: [char()] \n"
               " - The name of the csv file, excluding .csv, containing a previously found trip."
               " If missing, a linear route is generated as the starting trip. \n"
               "(Optional) -temp: [float(1)] \n" 
               " - The initial starting temperature. Default is 1000 \n"
               "(Optional) -decay: [float(1)]  \n"
               " - The decay rate for the annealing schedule. Default is .99 \n");
        return 1;
    }
    
	const char *tsp_name = concat(argv[1], ".tsp");
	coordinates *location_g;
	read_tsp(tsp_name);
    unsigned int N = meta->dim, *N_g;
	unsigned int i, kk;
	unsigned int *salesman_route = (unsigned int *)malloc((N + 1) * sizeof(unsigned int));

    // Get loss
    float T[2], *T_g;
	T[0] = 10000;
	T[1] = 10000;
	float decay = 0.99;
    // Get starting trip
	for (i = 0; i <= N; i++)
	    salesman_route[i] = i;
	// Set the starting and end points to be the same
	salesman_route[N] = salesman_route[0];
    
    
    
    // read in options
    for (int i = 1; i < argc; i++) {
        if (i + 1 != argc)    {
            if (strcmp(argv[i], "-trip=") == 0) {          
                const char *trip_name = concat(argv[i + 1], ".csv");
                read_trip(trip_name, salesman_route);    
            }
            if (strcmp(argv[i], "-temp=") == 0) {           
                // If atof cannot convert to a float, it returns 0
	            float user_temp = atof(argv[i + 1]);
	            if ( user_temp == 0){
	                printf("Error: Initial Temperature must be a non-zero number\n");
	                return 1;
	            }
	            T[0] = user_temp;
	            T[1] = T[0];
	        }
            if (strcmp(argv[i], "-decay=") == 0) {           
                // If atoi cannot convert to number, it returns 0
                float user_decay = atof(argv[i + 1]);
	            if (user_decay == 0){
	                printf("Error: Decay must be a number from 0 to 1\n");
	                return 1;
	            } else if (user_decay >= 1 || user_decay <= 0){
	                printf("Error: Decay must be a number from 0 to 1\n");
	                return 1;
	            } else {
	               decay = user_decay;
	            }    
            }
        }
    }
    
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
		original_loss += sqrt((location[salesman_route[i]].x - location[salesman_route[i + 1]].x) *
			(location[salesman_route[i]].x - location[salesman_route[i + 1]].x) +
			(location[salesman_route[i]].y - location[salesman_route[i + 1]].y) *
			(location[salesman_route[i]].y - location[salesman_route[i + 1]].y));
	}
	printf("Original Loss is:  %0.6f \n", original_loss);

	
    float optimized_loss_restart = original_loss;
    long int iter = 0;
    
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
	unsigned int *salesman_route_g, *salesman_route_2g, *salesman_route_restartg, *flag_g, *city_swap_one_g, *city_swap_two_g;
	int global_flag_h = 0, *global_flag_g;

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
	cudaMalloc((void**)&salesman_route_restartg, (N + 1) * sizeof(unsigned int));
	cudaCheckError();
	cudaMalloc((void**)&T_g, 2 * sizeof(float));
	cudaCheckError();
	cudaMalloc((void**)&flag_g, GRID_SIZE * sizeof(unsigned int));
	cudaCheckError();
	cudaMalloc((void**)&global_flag_g, sizeof(int));
	cudaCheckError();
	cudaMalloc((void**)&N_g, sizeof(unsigned int));
	cudaCheckError();


	cudaMemcpy(location_g, location, N * sizeof(coordinates), cudaMemcpyHostToDevice);
	cudaCheckError();
	cudaMemcpy(salesman_route_g, salesman_route, (N + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaCheckError();
	cudaMemcpy(salesman_route_2g, salesman_route, (N + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaCheckError();
	cudaMemcpy(salesman_route_restartg, salesman_route, (N + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaCheckError();
	cudaMemcpy(global_flag_g, &global_flag_h, sizeof(int), cudaMemcpyHostToDevice);
	cudaCheckError();
	cudaMemcpy(N_g, &N, sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaCheckError();

	// Number of thread blocks in grid
	// X is for the sampling, y is for manipulating the salesman's route
	dim3 blocksPerSampleGrid(GRID_SIZE / t_num, 1, 1);
	dim3 blocksPerTripGrid((N / t_num) + 1, 1, 1);
	dim3 threadsPerBlock(t_num, 1, 1);

	
	curandState_t* states;

	/* allocate space on the GPU for the random states */
	cudaMalloc((void**)&states, GRID_SIZE * sizeof(curandState_t));
	init <<<blocksPerSampleGrid, threadsPerBlock, 0 >>>(time(0), states);

	//time counter
	time_t t_start, t_end;
	t_start = time(NULL);
	
	// Var to check, if we have the same loss 10 times in a row, speed up decay
	int sames = 0;

    
	while (T[0] > 0.01/log(2*N))
	{
		// Copy memory from host to device
		cudaMemcpy(T_g, T, 2 * sizeof(float), cudaMemcpyHostToDevice);
		cudaCheckError();
		i = 1;
        kk = 1;
		while (i<1000){

			globalSwap <<<blocksPerSampleGrid, threadsPerBlock, 0 >>>(city_swap_one_g, city_swap_two_g,
				                                                      location_g, salesman_route_g,
				                                                      T_g, global_flag_g, N_g,
				                                                      states);  
			cudaCheckError();
			
			SwapUpdate <<<blocksPerSampleGrid, threadsPerBlock, 0 >>>(city_swap_one_g, city_swap_two_g,
				                                                      salesman_route_g, global_flag_g);
			cudaCheckError();
			
			localSwap <<<blocksPerSampleGrid, threadsPerBlock, 0 >>>(city_swap_one_g, city_swap_two_g,
				                                                     location_g, salesman_route_g,
				                                                     T_g, global_flag_g, N_g,
				                                                     states); 
			cudaCheckError();
			
			SwapUpdate <<<blocksPerSampleGrid, threadsPerBlock, 0 >>>(city_swap_one_g, city_swap_two_g,
				                                                      salesman_route_g, global_flag_g);
			cudaCheckError();
		/*
			InsertionUpdateTrip <<<blocksPerTripGrid, threadsPerBlock, 0 >>>(salesman_route_g, salesman_route_2g, N_g);
			cudaCheckError();
			globalInsertion <<<blocksPerSampleGrid, threadsPerBlock, 0 >>>(city_swap_one_g, city_swap_two_g,
				                                                           location_g, salesman_route_g,
				                                                           T_g, global_flag_g, N_g,
				                                                           states); 
			cudaCheckError();
			
			InsertionUpdate <<<blocksPerTripGrid, threadsPerBlock, 0 >>>(city_swap_one_g, city_swap_two_g,
				                                                         salesman_route_g, salesman_route_2g,
				                                                         global_flag_g);
		    cudaCheckError();
            InsertionUpdateEnd <<<blocksPerTripGrid, threadsPerBlock, 0 >>>(city_swap_one_g, city_swap_two_g,
				                                                         salesman_route_g, salesman_route_2g,
				                                                         global_flag_g);
            cudaCheckError();
            
			InsertionUpdateTrip <<<blocksPerTripGrid, threadsPerBlock, 0 >>>(salesman_route_g, salesman_route_2g, N_g);
			cudaCheckError();
			
			localInsertion <<<blocksPerSampleGrid, threadsPerBlock, 0 >>>(city_swap_one_g, city_swap_two_g,
				                                                          location_g, salesman_route_g,
				                                                          T_g, global_flag_g, N_g,
				                                                          states);
			cudaCheckError();
			
			//InsertionUpdateTrip <<<blocksPerTripGrid, threadsPerBlock, 0 >>>(salesman_route_g, salesman_route_2g, N_g);
			//cudaCheckError();
			
			InsertionUpdate <<<blocksPerTripGrid, threadsPerBlock, 0 >>>(city_swap_one_g, city_swap_two_g,
				                                                          salesman_route_g, salesman_route_2g,
				                                                          global_flag_g);
			cudaCheckError();
			
					    
            InsertionUpdateEnd <<<blocksPerTripGrid, threadsPerBlock, 0 >>>(city_swap_one_g, city_swap_two_g,
				                                                         salesman_route_g, salesman_route_2g,
				                                                         global_flag_g);
			cudaCheckError();
			*/
				                                                      
			// We put more emphasis on 2-opt routes
			 while(kk < 1000){

                 global2Opt<<<blocksPerSampleGrid, threadsPerBlock, 0 >>>(city_swap_one_g, city_swap_two_g,
				                                                          location_g, salesman_route_g,
                                                                          T_g, global_flag_g, N_g,
                                                                          states);
                 //cudaCheckError();
                
                 InsertionUpdateTrip <<<blocksPerTripGrid, threadsPerBlock, 0 >>>(salesman_route_g, salesman_route_2g, N_g);
                 cudaCheckError();
                 Opt2Update<<<blocksPerTripGrid, threadsPerBlock, 0 >>>(city_swap_one_g, city_swap_two_g,
                                                                        salesman_route_g, salesman_route_2g,
                                                                        global_flag_g);
                 //cudaCheckError();
                 local2Opt<<<blocksPerSampleGrid, threadsPerBlock, 0 >>>(city_swap_one_g, city_swap_two_g,
				                                                          location_g, salesman_route_g,
                                                                          T_g, global_flag_g, N_g,
                                                                          states);
                 //cudaCheckError();
                
                 InsertionUpdateTrip <<<blocksPerTripGrid, threadsPerBlock, 0 >>>(salesman_route_g, salesman_route_2g, N_g);
                 cudaCheckError();
                 Opt2Update<<<blocksPerTripGrid, threadsPerBlock, 0 >>>(city_swap_one_g, city_swap_two_g,
                                                                        salesman_route_g, salesman_route_2g,
                                                                        global_flag_g);
                 //cudaCheckError();
                 
			     kk++;
			 }
			
			i++;
		}
		cudaMemcpy(salesman_route, salesman_route_g, (N + 1) * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		cudaCheckError();
		float optimized_loss = 0;
		for (i = 0; i < N; i++){
			optimized_loss += sqrt((location[salesman_route[i]].x - location[salesman_route[i + 1]].x) *
				(location[salesman_route[i]].x - location[salesman_route[i + 1]].x) +
				(location[salesman_route[i]].y - location[salesman_route[i + 1]].y) *
				(location[salesman_route[i]].y - location[salesman_route[i + 1]].y));
		}
		printf("| Loss: %.6f | Temp: %f | Iter: %ld |\n", optimized_loss, T[0], iter);
		T[0] = T[0] * decay;
		
		iter++;
		// This grabs the best trip overall
		if (optimized_loss < optimized_loss_restart){
		    optimized_loss_restart = optimized_loss;
		    InsertionUpdateTrip <<<blocksPerTripGrid, threadsPerBlock, 0 >>>(salesman_route_g, salesman_route_restartg, N_g);
			cudaCheckError();
			sames = 0;
	    } else if (optimized_loss == optimized_loss_restart){
	        sames++;
	        if (sames > 10){
	            T[0] = T[0] * 0.8;
	            }
	    }
	}
	//print time spent
	t_end = time(NULL);
	printf("time = %f\n", difftime(t_end, t_start));

	cudaMemcpy(salesman_route, salesman_route_g, (N + 1) * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaCheckError();
	
	// We have to redefine optimized loss for some reason?
	float optimized_loss = 0;
	for (i = 0; i < N; i++){
		optimized_loss += sqrt((location[salesman_route[i]].x - location[salesman_route[i + 1]].x) *
			(location[salesman_route[i]].x - location[salesman_route[i + 1]].x) +
			(location[salesman_route[i]].y - location[salesman_route[i + 1]].y) *
			(location[salesman_route[i]].y - location[salesman_route[i + 1]].y));
	}
	
	// If it's worse than the restart make the route the restart.
	if (optimized_loss > optimized_loss_restart){
        cudaMemcpy(salesman_route, salesman_route_restartg, (N + 1) * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaCheckError();
	}
	
	optimized_loss = 0;
	for (i = 0; i < N; i++){
		optimized_loss += sqrt((location[salesman_route[i]].x - location[salesman_route[i + 1]].x) *
			(location[salesman_route[i]].x - location[salesman_route[i + 1]].x) +
			(location[salesman_route[i]].y - location[salesman_route[i + 1]].y) *
			(location[salesman_route[i]].y - location[salesman_route[i + 1]].y));
	}
	
	printf("Original Loss is:  %0.6f \n", original_loss); 
	printf("Optimized Loss is: %.6f \n", optimized_loss);

	// Write the best trip to CSV
	FILE *best_trip;
	const char *filename = concat(argv[1],"_trip.csv");
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
	cudaFree(salesman_route_restartg);
	cudaCheckError();
	free(salesman_route);
	free(city_swap_one_h);
	free(city_swap_two_h);
	free(flag_h);
	free(location);
	getchar();
	return 0;
}


