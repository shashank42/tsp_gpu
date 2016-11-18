#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>
#include <string.h>
#include <limits.h>
#include <assert.h>
#include <unistd.h>
#include "utils.h"
#include "tsp_solve.h"
#include <iostream>
#include <fstream>

#define t_num 1024
#define GRID_SIZE 1024
#define LINE_BUF_LEN 100

/*************************************************** 
* Desc: Simulated Annealing for roundtrips
* Notes:
*  For more samples define GRID_SIZE as a multiple of t_num such as 512000, 2048000, or the (max - 1024) grid size 2147482623
*  Some compliation options that can speed things up
*  --use_fast_math 
* --optimize=5
*  --gpu-architecture=compute_35
*  I use something like
*   nvcc --optimize=5 --use_fast_math -arch=compute_35 tsp_cuda.cu -o tsp_cuda
*
*  We are going to try some stuff for temp from this adaptive simulated annealing paper
*  https://arxiv.org/pdf/cs/0001018.pdf
****************************************************/

/***************************************************************************
*   Variable Descriptions:
* tsp_name: [const char()]
*  - The name of the tsp file we get our meta data and location data from
* N, N_g: [integer(1)]
*  - The number of cities. We also place this on the device for checking corner cases
* i,j,m [integer(1)]
*  - Standard iters
* salesman_route [unsigned integer(N + 1)]
*  - The route that we will traverse over, starts and ends at the same location
* city_swap_one_h/g: [integer(t_num)]
*  - Host/Device memory for city one
* city_swap_two_h/g: [integer(t_num)]
*  - Host/Device memory for city two
* flag_h/g: [integer(t_num)]
*  - Host/Device memory for flag of accepted step
* salesman_route_g: [integer(N)]
*  - Device memory for the salesmans route
* starting loss [float(1)]
*  - The loss from the naive route
* T,T_start,T_g [float(1)]
*  - T is the temperature that is changed as we iterate through SA, T_start is the initial temperature
*  - T_g is device memory for the temperature within SA
* r_h/g:  [float(t_num)]
*  - host/device memory for the random number when deciding acceptance
* flag_h/g: [integer(t_num)]
*  - host/device memory for acceptance vector
* original_loss_h/g: [integer(1)]
*  - The host/device current loss function
* new_loss_h/g: [integer(t_num)]
*  - The host/device memory for the proposal loss function
* 
*****************************************************************************/
 int main(){
 
     // Set up the structs, location is allocated in read_tsp
     // meta and location come from utils.h
     const char *tsp_name = "dsj1000.tsp";
     read_tsp(tsp_name);
     unsigned int N = meta -> dim, *N_g;     
     // start counters for cities
     unsigned int i, j, m;
     unsigned int *salesman_route = (unsigned int *)malloc((N + 1) * sizeof(unsigned int));
     float original_loss = 0;
     float starting_loss = original_loss;
     float T = 9,T_start = 9, *T_g, *r_g;
     float *r_h = (float *)malloc(GRID_SIZE * sizeof(float));
     float new_loss_h = 0;
     float swap_distance;
     float iter = 1.0;
     unsigned int *city_swap_one_h = (unsigned int *)malloc(GRID_SIZE * sizeof(unsigned int));
     unsigned int *city_swap_two_h = (unsigned int *)malloc(GRID_SIZE * sizeof(unsigned int));
     unsigned int *flag_h = (unsigned int *)malloc(GRID_SIZE * sizeof(unsigned int));
     unsigned int *city_swap_one_g, *city_swap_two_g, *salesman_route_g, *flag_g;
     unsigned int global_flag_h = 0, *global_flag_g;
     coordinates *location_g;
     // Create space on the device for each var
     cudaError_t err = cudaMalloc((void**)&city_swap_one_g, GRID_SIZE * sizeof(unsigned int));
     //printf("\n Cuda malloc city swap one: %s \n", cudaGetErrorString(err));
     cudaMalloc((void**)&city_swap_two_g, GRID_SIZE * sizeof(unsigned int));
     cudaMalloc((void**)&location_g, N * sizeof(coordinates));
     cudaMalloc((void**)&salesman_route_g, (N + 1) * sizeof(unsigned int));
     cudaMalloc((void**)&T_g, sizeof(float));
     cudaMalloc((void**)&r_g, GRID_SIZE * sizeof(float));
     cudaMalloc((void**)&flag_g, GRID_SIZE * sizeof(unsigned int));
     cudaMalloc((void**)&global_flag_g, sizeof(unsigned int));
     cudaMalloc((void**)&N_g, sizeof(unsigned int));
     // Copy the city locations to device as well as the number of cities
     cudaMemcpy(location_g, location, N * sizeof(coordinates), cudaMemcpyHostToDevice);
     cudaMemcpy(N_g, &N, sizeof(unsigned int), cudaMemcpyHostToDevice);

     // just make one inital guess route, a simple linear path
     for (i = 0; i <= N; i++)
         salesman_route[i] = i;
         
     // Set the starting and end points to be the same
     salesman_route[N] = salesman_route[0];
    

     // Calculate the original loss
     for (i = 0; i < N; i++){
         original_loss += (location[salesman_route[i]].x - location[salesman_route[i+1]].x) *
                          (location[salesman_route[i]].x - location[salesman_route[i+1]].x) +
                          (location[salesman_route[i]].y - location[salesman_route[i+1]].y) *
                          (location[salesman_route[i]].y - location[salesman_route[i+1]].y);
     }
     starting_loss = original_loss;
     printf("Original Loss is: %.6f \n", original_loss);
     printf("Number of cities: %d \n", N); 
     //Best for 100,000: The starting loss was 33,346,203,648 and the final loss was 10,243,860,480 
     while (T > 1){
         // Init parameters
         global_flag_h = 0;
         for(m = 0; m < GRID_SIZE; m++){
             // pick first city to swap
             // Notice that we travel along N + 1 cities, but only select between 0 and N
             // Since the first and last stop have to be the same we only look 0
             city_swap_one_h[m] = rand_interval(0, N-2);
             /* swap_distance defines how far the second city can be from the first
                 This sort of gives us a nice slope
                 http://www.wolframalpha.com/input/?i=e%5E(-+sqrt(ln(9999%2Ft))))+from+9999+to+0
             */
             swap_distance = exp(-sqrt(log(T_start / T)));
             j = (unsigned int)floor(1 + city_swap_one_h[m] * swap_distance); 
             // pick second city to swap
             city_swap_two_h[m] = (city_swap_one_h[m] + j) % (N-1);
             // Check we are not at the first or last city for city two
             if (city_swap_two_h[m] == 0)
               city_swap_two_h[m] += 1;
             if (city_swap_two_h[m] == N - 1)
               city_swap_two_h[m] -= 1;
             r_h[m] = (float)rand() / (float)RAND_MAX ;
             
             //set our flags and new loss to 0
             flag_h[m] = 0;
             //printf("Iteration: %d \n City one is: %d and city Two is: %d \n",m, city_swap_one_h[m], city_swap_two_h[m]);
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
          dim3 blocksPerGrid(GRID_SIZE/t_num,1,1);
          dim3 threadsPerBlock(t_num,1,1);
    
          tsp<<<blocksPerGrid, threadsPerBlock, 0>>>(city_swap_one_g, city_swap_two_g,
                                                         location_g, salesman_route_g,
                                                         T_g, r_g, flag_g, global_flag_g,
                                                         N_g);

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
                } else {
                    // switch the two cities that led to an accepted proposal
                    unsigned int tmp = salesman_route[city_swap_one_h[i]];
                    salesman_route[city_swap_one_h[i]] = salesman_route[city_swap_two_h[i]];
                    salesman_route[city_swap_two_h[i]] = tmp;
                    if (tmp == 0)
                      salesman_route[N] = tmp;
                      
                    new_loss_h = 0;
                    for (i = 0; i < N - 1; i++){
                      new_loss_h += (location[salesman_route[i]].x - location[salesman_route[i+1]].x) *
                                    (location[salesman_route[i]].x - location[salesman_route[i+1]].x) +
                                    (location[salesman_route[i]].y - location[salesman_route[i+1]].y) *
                                    (location[salesman_route[i]].y - location[salesman_route[i+1]].y);
                    }

                    // set old loss function to new
                    original_loss = new_loss_h;
                    //decrease temp

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
     
     if ((int)iter % 500 == 0){
         printf(" Current Temperature is %.6f \n", T);
         printf("\n Current Loss is: %.6f \n", original_loss);
         printf("\n Current Iteration is: %.6f \n", iter);
     }
     //T = 1;
     T = T_start /log(iter);
     iter += 1.0f;
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
     cudaFree(N_g);
     free(salesman_route);
     free(city_swap_one_h);
     free(city_swap_two_h);
     free(flag_h);
     free(location);
     free(meta);
     return 0;
}
             
         
         
         

