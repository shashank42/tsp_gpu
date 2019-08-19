/*
    columbus: Software for computing approximate solutions to the traveling
   salesman's problem on GPUs Copyright (C) 2016 Steve Bronder and Haoyan Min

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

#include <assert.h>
#include <ctype.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <fstream>
#include <iostream>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// If NDEBUG is defined, cudaCheckError() will be empty
#define NDEBUG
#include "kernels/utils.h"
#include "kernels/initialize_rng.h"
#include "kernels/insert_sampler.h"
#include "kernels/opt2_sampler.h"
#include "kernels/swap_sampler.h"


#define t_num 1024
#define GRID_SIZE 131072

/*
For more samples define GRID_SIZE as a multiple of t_num such as 512000,
2048000, or the (max - 1024) grid size 2147482623 A good grid size is the number
of SM's you have times the number of blocks each can take in times max threads
per block I have 8 cores that can hold 16 blocks of 1024 cores so my best is
131072 Some compiler options that can speed things up
--use_fast_math
--optimize=5
--gpu-architecture=compute_35
I use something like
NOTE: You need to use the -lcurand flag to compile for the RNG.
nvcc --optimize=5 --use_fast_math -arch=compute_35 main_solver.cu -o tsp_cuda
-lcurand
*/

int main(int argc, char *argv[]) {

  // Reading in inputs
  if (argc == 1) {
    printf(
        "Inputs: \n"
        "(Required) input_file.tsp: [char()] \n"
        " - The name of the tsp file, excluding .tsp at the end, containing "
        "the cities to travel over. \n"
        "(Optional) -trip: [char()] \n"
        " - The name of the csv file, excluding .csv, containing a previously "
        "found trip."
        " If missing, a linear route is generated as the starting trip. \n"
        "(Optional) -temp: [float(1)] \n"
        " - The initial starting temperature. Default is 1000 \n"
        "(Optional) -decay: [float(1)]  \n"
        " - The decay rate for the annealing schedule. Default is .99 \n"
        "(Optional) -maxiter: [integer(1)]  \n"
        " - The maximum number of iterations until failure. \n"
        "  Default is -1, which runs until temperature goes to the minimum.\n"
        "(Optional) -global_search: [float(1)]  \n"
        " - A parameter that controls the variance of the second city search "
        "space,\n"
        "   such that the variance is [30 + exp(global_search/Temp) * N]. "
        "default is .01.\n"
        "  See An example of what this controls here:\n"
        "(Optional) -local_search: [float(1)]  \n"
        " - A parameter that controls the variance of the second city search "
        "space,\n"
        "   such that the variance is [30 + exp(local_search/Temp) * N]. "
        "default is 1.\n");

    return 1;
  }

  const char *tsp_name = concat(argv[1], ".tsp");
  coordinates *location_g;
  read_tsp(tsp_name);
  unsigned int N = meta->dim, *N_g;
  unsigned int i;
  unsigned int *salesman_route =
      (unsigned int *)malloc((N + 1) * sizeof(unsigned int));
  float sample_area_local, sample_area_global, *sample_area_local_g,
      *sample_area_global_g;
  sample_area_global = 0.01;
  sample_area_local = 1;
  // just make one inital guess route, a simple linear path
  for (i = 0; i <= N; i++)
    salesman_route[i] = i;

  // Set the starting and end points to be the same
  salesman_route[N] = salesman_route[0];

  // Get loss
  float T[1], *T_g;
  T[1] = .03;
  float decay = 0.99;
  int maxiter = -1;
  // Get starting trip
  for (i = 0; i <= N; i++)
    salesman_route[i] = i;
  // Set the starting and end points to be the same
  salesman_route[N] = salesman_route[0];

  // read in options
  for (int i = 1; i < argc; i++) {
    if (i + 1 != argc) {
      if (strcmp(argv[i], "-trip=") == 0) {
        const char *trip_name = concat(argv[i + 1], ".csv");
        read_trip(trip_name, salesman_route);
      }
      if (strcmp(argv[i], "-temp=") == 0) {
        // If atof cannot convert to a float, it returns 0
        float user_temp = atof(argv[i + 1]);
        if (user_temp == 0) {
          printf("Error: Initial Temperature must be a non-zero number\n");
          return 1;
        }
        T[0] = user_temp;
        T[1] = T[0];
      }
      if (strcmp(argv[i], "-maxiter=") == 0) {
        // If atof cannot convert to a float, it returns 0
        float user_iter = atoi(argv[i + 1]);
        if (user_iter == 0) {
          printf("Error: max iter cannot be zero\n");
          return 1;
        }
        maxiter = user_iter;
      }
      if (strcmp(argv[i], "-decay=") == 0) {
        // If atoi cannot convert to number, it returns 0
        float user_decay = atof(argv[i + 1]);
        if (user_decay == 0) {
          printf("Error: Decay must be a number from 0 to 1\n");
          return 1;
        } else if (user_decay >= 1 || user_decay <= 0) {
          printf("Error: Decay must be a number from 0 to 1\n");
          return 1;
        } else {
          decay = user_decay;
        }
      }
      if (strcmp(argv[i], "-global_search=") == 0) {
        // If atoi cannot convert to number, it returns 0
        float user_global = atof(argv[i + 1]);
        if (user_global == 0) {
          printf("Error: global search param must be greater than 0. \n");
          return 1;
        } else {
          sample_area_global = user_global;
        }
      }
      if (strcmp(argv[i], "-local_search=") == 0) {
        // If atoi cannot convert to number, it returns 0
        float user_local = atof(argv[i + 1]);
        if (user_local == 0) {
          printf("Error: local search param must be greater than 0. \n");
          return 1;
        } else {
          sample_area_local = user_local;
        }
      }
    }
  }

  // Calculate the original loss
  float original_loss = 0;
  for (i = 0; i < N; i++) {
    original_loss += sqrtf(
        (location[salesman_route[i]].x - location[salesman_route[i + 1]].x) *
            (location[salesman_route[i]].x -
             location[salesman_route[i + 1]].x) +
        (location[salesman_route[i]].y - location[salesman_route[i + 1]].y) *
            (location[salesman_route[i]].y -
             location[salesman_route[i + 1]].y));
  }
  printf("Original Loss is:  %0.6f \n", original_loss);
  float optimized_loss_restart = original_loss;
  // Keep the original loss for comparison pre/post algorithm
  // SET THE LOSS HERE

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
  unsigned int *city_swap_one_h =
      (unsigned int *)malloc(GRID_SIZE * sizeof(unsigned int));
  unsigned int *city_swap_two_h =
      (unsigned int *)malloc(GRID_SIZE * sizeof(unsigned int));
  unsigned int *flag_h =
      (unsigned int *)malloc(GRID_SIZE * sizeof(unsigned int));
  unsigned int *salesman_route_g, *salesman_route_2g, *salesman_route_restartg,
      *flag_g, *city_swap_one_g, *city_swap_two_g;
  int global_flag_h = -1, *global_flag_g;

  cudaMalloc((void **)&city_swap_one_g, GRID_SIZE * sizeof(unsigned int));
  cudaCheckError();
  cudaMalloc((void **)&city_swap_two_g, GRID_SIZE * sizeof(unsigned int));
  cudaCheckError();
  cudaMalloc((void **)&location_g, N * sizeof(coordinates));
  cudaCheckError();
  cudaMalloc((void **)&salesman_route_g, (N + 1) * sizeof(unsigned int));
  cudaCheckError();
  cudaMalloc((void **)&salesman_route_2g, (N + 1) * sizeof(unsigned int));
  cudaCheckError();
  cudaMalloc((void **)&salesman_route_restartg, (N + 1) * sizeof(unsigned int));
  cudaCheckError();
  cudaMalloc((void **)&T_g, sizeof(float));
  cudaCheckError();
  cudaMalloc((void **)&sample_area_global_g, sizeof(float));
  cudaCheckError();
  cudaMalloc((void **)&sample_area_local_g, sizeof(float));
  cudaCheckError();
  cudaMalloc((void **)&flag_g, GRID_SIZE * sizeof(int));
  cudaCheckError();
  cudaMalloc((void **)&global_flag_g, sizeof(int));
  cudaCheckError();
  cudaMalloc((void **)&N_g, sizeof(unsigned int));
  cudaCheckError();

  cudaMemcpy(location_g, location, N * sizeof(coordinates),
             cudaMemcpyHostToDevice);
  cudaCheckError();
  cudaMemcpy(salesman_route_g, salesman_route, (N + 1) * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  cudaCheckError();
  cudaMemcpy(salesman_route_2g, salesman_route, (N + 1) * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  cudaCheckError();
  cudaMemcpy(salesman_route_restartg, salesman_route,
             (N + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaCheckError();
  cudaMemcpy(global_flag_g, &global_flag_h, sizeof(int),
             cudaMemcpyHostToDevice);
  cudaCheckError();
  cudaMemcpy(N_g, &N, sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaCheckError();
  cudaMemcpy(sample_area_global_g, &sample_area_global, sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  cudaCheckError();
  cudaMemcpy(sample_area_local_g, &sample_area_local, sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  cudaCheckError();
  // Beta is the decay rate
  // float beta = 0.0001;
  // We are going to try some stuff for temp from this adaptive simulated
  // annealing paper https://arxiv.org/pdf/cs/0001018.pdf

  // Number of thread blocks in grid
  // X is for the sampling, y is for manipulating the salesman's route
  dim3 blocksPerSampleGrid(GRID_SIZE / t_num, 1, 1);
  dim3 blocksPerTripGrid((N / t_num) + 1, 1, 1);
  dim3 threadsPerBlock(t_num, 1, 1);

  // Trying out random gen in cuda
  curandState_t *states;

  /* allocate space on the GPU for the random states */
  cudaMalloc((void **)&states, GRID_SIZE * sizeof(curandState_t));
  init<<<blocksPerSampleGrid, threadsPerBlock, 0>>>(time(0), states);

  // time counter
  time_t t_start, t_end;
  t_start = time(NULL);
  long int iter = 1;
  printf("Ending Temp: %f \n", FLT_EPSILON * 100);
  // int sames = 0;
  printf(" Loss | Temp | Iter | Time \n");
  while (T[0] > FLT_EPSILON * 100 | T[0] == 0.0) {
    // Copy memory from host to device
    cudaMemcpy(T_g, T, sizeof(float), cudaMemcpyHostToDevice);
    i = 1;

    while (i < 2000) { // key

      // two opt
      cudaMemcpy(salesman_route_2g, salesman_route_g,
                 (N + 1) * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
      cudaCheckError();

      cudaMemcpy(global_flag_g, &global_flag_h, sizeof(int),
                 cudaMemcpyHostToDevice);
      cudaCheckError();

      twoOptStep<<<blocksPerSampleGrid, threadsPerBlock, 0>>>(
          city_swap_one_g, city_swap_two_g, location_g, salesman_route_g, T_g,
          global_flag_g, N_g, states, sample_area_global_g);
      cudaCheckError();

      opt2Update<<<blocksPerTripGrid, threadsPerBlock, 0>>>(
          city_swap_one_g, city_swap_two_g, salesman_route_g, salesman_route_2g,
          global_flag_g);
      cudaCheckError();

      cudaMemcpy(salesman_route_2g, salesman_route_g,
                 (N + 1) * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
      cudaCheckError();

      cudaMemcpy(global_flag_g, &global_flag_h, sizeof(int),
                 cudaMemcpyHostToDevice);
      cudaCheckError();

      twoOptStep<<<blocksPerSampleGrid, threadsPerBlock, 0>>>(
          city_swap_one_g, city_swap_two_g, location_g, salesman_route_g, T_g,
          global_flag_g, N_g, states, sample_area_local_g);
      cudaCheckError();

      opt2Update<<<blocksPerTripGrid, threadsPerBlock, 0>>>(
          city_swap_one_g, city_swap_two_g, salesman_route_g, salesman_route_2g,
          global_flag_g);
      cudaCheckError();

      // insertionstep
      cudaMemcpy(salesman_route_2g, salesman_route_g,
                 (N + 1) * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
      cudaCheckError();

      cudaMemcpy(global_flag_g, &global_flag_h, sizeof(int),
                 cudaMemcpyHostToDevice);
      cudaCheckError();

      insertionStep<<<blocksPerSampleGrid, threadsPerBlock, 0>>>(
          city_swap_one_g, city_swap_two_g, location_g, salesman_route_g, T_g,
          global_flag_g, N_g, states, sample_area_global_g);
      cudaCheckError();

      insertionUpdate<<<blocksPerTripGrid, threadsPerBlock, 0>>>(
          city_swap_one_g, city_swap_two_g, salesman_route_g, salesman_route_2g,
          global_flag_g);
      cudaCheckError();

      insertionUpdateEndPoints<<<blocksPerTripGrid, threadsPerBlock, 0>>>(
          city_swap_one_g, city_swap_two_g, salesman_route_g, salesman_route_2g,
          global_flag_g);
      cudaCheckError();

      cudaMemcpy(salesman_route_2g, salesman_route_g,
                 (N + 1) * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
      cudaCheckError();

      cudaMemcpy(global_flag_g, &global_flag_h, sizeof(int),
                 cudaMemcpyHostToDevice);
      cudaCheckError();

      insertionStep<<<blocksPerSampleGrid, threadsPerBlock, 0>>>(
          city_swap_one_g, city_swap_two_g, location_g, salesman_route_g, T_g,
          global_flag_g, N_g, states, sample_area_local_g);
      cudaCheckError();

      insertionUpdate<<<blocksPerTripGrid, threadsPerBlock, 0>>>(
          city_swap_one_g, city_swap_two_g, salesman_route_g, salesman_route_2g,
          global_flag_g);
      cudaCheckError();

      insertionUpdateEndPoints<<<blocksPerTripGrid, threadsPerBlock, 0>>>(
          city_swap_one_g, city_swap_two_g, salesman_route_g, salesman_route_2g,
          global_flag_g);
      cudaCheckError();

      cudaMemcpy(global_flag_g, &global_flag_h, sizeof(int),
                 cudaMemcpyHostToDevice);
      cudaCheckError();
      // swap step

      cudaMemcpy(salesman_route_2g, salesman_route_g,
                 (N + 1) * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
      cudaCheckError();

      swapStep<<<blocksPerSampleGrid, threadsPerBlock, 0>>>(
          city_swap_one_g, city_swap_two_g, location_g, salesman_route_g, T_g,
          global_flag_g, N_g, states, sample_area_global_g);
      cudaCheckError();

      swapUpdate<<<blocksPerSampleGrid, threadsPerBlock, 0>>>(
          city_swap_one_g, city_swap_two_g, salesman_route_g, salesman_route_2g,
          global_flag_g);
      cudaCheckError();

      cudaMemcpy(salesman_route_2g, salesman_route_g,
                 (N + 1) * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
      cudaCheckError();

      cudaMemcpy(global_flag_g, &global_flag_h, sizeof(int),
                 cudaMemcpyHostToDevice);
      cudaCheckError();

      swapStep<<<blocksPerSampleGrid, threadsPerBlock, 0>>>(
          city_swap_one_g, city_swap_two_g, location_g, salesman_route_g, T_g,
          global_flag_g, N_g, states, sample_area_local_g);
      cudaCheckError();

      swapUpdate<<<blocksPerSampleGrid, threadsPerBlock, 0>>>(
          city_swap_one_g, city_swap_two_g, salesman_route_g, salesman_route_2g,
          global_flag_g);
      cudaCheckError();

      cudaMemcpy(global_flag_g, &global_flag_h, sizeof(int),
                 cudaMemcpyHostToDevice);
      cudaCheckError();

      cudaMemcpy(salesman_route_2g, salesman_route_g,
                 (N + 1) * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
      cudaCheckError();

      i++;
    }
    cudaMemcpy(salesman_route, salesman_route_g, (N + 1) * sizeof(unsigned int),
               cudaMemcpyDeviceToHost);
    cudaCheckError();
    float optimized_loss = 0;
    for (i = 0; i < N; i++) {
      optimized_loss += sqrt(
          (location[salesman_route[i]].x - location[salesman_route[i + 1]].x) *
              (location[salesman_route[i]].x -
               location[salesman_route[i + 1]].x) +
          (location[salesman_route[i]].y - location[salesman_route[i + 1]].y) *
              (location[salesman_route[i]].y -
               location[salesman_route[i + 1]].y));
    }
    printf(" %.6f | %f | %ld | %f\n", optimized_loss, T[0], iter,
           difftime(time(NULL), t_start));
    T[0] = T[0] * decay;
    iter++;
    // This grabs the best trip overall
    if (optimized_loss < optimized_loss_restart) {
      optimized_loss_restart = optimized_loss;
      cudaMemcpy(salesman_route_restartg, salesman_route_g,
                 (N + 1) * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
      cudaCheckError();
      // sames = 0;
    } /*else if (abs(optimized_loss - optimized_loss_restart) < 2){
    // If we are only gaining by one then we can start speeding things up
        sames++;
        if (sames > 10){
            T[0] = T[0] * 0.8;
            }
    }*/
    if (maxiter > 0 && maxiter < iter)
      break;
  }

  t_end = time(NULL);
  printf("time = %f\n", difftime(t_end, t_start));

  cudaMemcpy(salesman_route, salesman_route_g, (N + 1) * sizeof(unsigned int),
             cudaMemcpyDeviceToHost);
  cudaCheckError();

  // We have to redefine optimized loss for some reason?
  float optimized_loss = 0;
  for (i = 0; i < N; i++) {
    optimized_loss += sqrt(
        (location[salesman_route[i]].x - location[salesman_route[i + 1]].x) *
            (location[salesman_route[i]].x -
             location[salesman_route[i + 1]].x) +
        (location[salesman_route[i]].y - location[salesman_route[i + 1]].y) *
            (location[salesman_route[i]].y -
             location[salesman_route[i + 1]].y));
  }

  // If it's worse than the restart make the route the restart.
  if (optimized_loss > optimized_loss_restart) {
    cudaMemcpy(salesman_route, salesman_route_restartg,
               (N + 1) * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaCheckError();
  }

  optimized_loss = 0;
  for (i = 0; i < N; i++) {
    optimized_loss += sqrt(
        (location[salesman_route[i]].x - location[salesman_route[i + 1]].x) *
            (location[salesman_route[i]].x -
             location[salesman_route[i + 1]].x) +
        (location[salesman_route[i]].y - location[salesman_route[i + 1]].y) *
            (location[salesman_route[i]].y -
             location[salesman_route[i + 1]].y));
  }

  printf("Original Loss is:  %0.6f \n", original_loss);
  printf("Optimized Loss is: %.6f \n", optimized_loss);

  // Write the best trip to CSV
  FILE *best_trip;
  const char *filename = concat(argv[1], "_trip.csv");
  best_trip = fopen(filename, "w+");
  fprintf(best_trip, "location,coordinate_x,coordinate_y\n");
  for (i = 0; i < N + 1; i++) {
    fprintf(best_trip, "%d,%.6f,%.6f\n", salesman_route[i],
            location[salesman_route[i]].x, location[salesman_route[i]].y);
  }
  fclose(best_trip);

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
  cudaFree(global_flag_g);
  cudaCheckError();
  cudaFree(salesman_route_restartg);
  cudaCheckError();
  cudaFree(sample_area_global_g);
  cudaCheckError();
  cudaFree(sample_area_local_g);
  cudaCheckError();
  free(salesman_route);
  free(city_swap_one_h);
  free(city_swap_two_h);
  free(flag_h);
  free(location);
  return 0;
}
