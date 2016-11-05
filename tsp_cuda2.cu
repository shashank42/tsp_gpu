#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>

#define t_num 256
#define N 100

/* BEGIN KERNEL
Input:
- i: A vector of cities to swap for the first swap choice
- k: A vector of cities to swap for the second swap choice
- dist: The distance matrix of each city
- salesman_route: The route the salesman will travel
- T: The current temperature
- r: The random number to compare against for S.A.
*/
__global__ static void tsp(int* i, int* k, float *dist, int *salesman_route,
                           float *T, float *r, int *flag){
    
    const int tid = threadIdx.x;
    float delta, p, b = 1;
    
    // first city to swap
    int salesman_route_i = salesman_route[i[tid]];
    int salesman_route_iminus_mod = salesman_route[(i[tid] - 1 + N) % N];
    int salesman_route_iplus_mod  = salesman_route[(i[tid] + 1) % N];
    
    // second city to swap
    int salesman_route_k = salesman_route[k[tid]];
    int salesman_route_kplus_mod  = salesman_route[k[tid] + 1 % N];
    int salesman_route_kminus_mod = salesman_route[(k[tid] - 1 + N) % N];
    
    // we should return this so we know the minimum route -S.
    delta = dist[salesman_route_iminus_mod * N + salesman_route_k] +
            dist[salesman_route_k * N + salesman_route_iplus_mod]  +
            dist[salesman_route_kminus_mod * N + salesman_route_i] +
            dist[salesman_route_i * N + salesman_route_kplus_mod]  -
            dist[salesman_route_iminus_mod * N + salesman_route_i] - 
            dist[salesman_route_i * N + salesman_route_iplus_mod] -
            dist[salesman_route_kminus_mod * N + salesman_route_k] - 
            dist[salesman_route_k * N + salesman_route_kplus_mod];
    p = exp(-delta * b / T[0]);
    if (p > r[tid])
      flag[tid] = 1;
    else
      flag[tid] = 0;
 }
 // END KERNEL
 
 int main(){
 
     // start counters for cities
     int i, j, m;
     
     // city's x y coordinates
     struct coordinates {
         int x;
         int y;
     };
     
     struct coordinates location[N];
     
     // the order of the salesman problem
     int salesman_route[N];
     
     // initialize the coordinates and sequence
     for(i = 0; i < N; i++){
         location[i].x = rand() % 1000;
         location[i].y = rand() % 1000;
     }
     
     // distance
     float dist[N * N];
     
     for(i = 0; i < N; i++){
         for (j = 0; j < N; j++){
             // Calculate the euclidian distance between each city
             // use pow() here instead?
             dist[i * N + j] = (location[i].x - location[j].x) * (location[i].x - location[j].x) +
                               (location[j].y - location[j].y) * (location[i].y - location[j].y);
         }
     }
     
     float dist_g[N * N], T = 5, T_g[1], r_h[t_num], r_g[t_num];
     
     /*
     Defining device variables:
     i_h/g: Host/Device memory for city one
     k_h/g: Host/Device memory for city two
     flag_h/g: Host/Device memory for flag of accepted step
     salesman_route_g: Device memory for the salesmans route
     r_g is the random number for deciding acceptance
     flag is the acceptance vector
     */
     int i_h[t_num], i_g[t_num],    k_h[t_num], k_g[t_num],
       salesman_route_g[N],  flag_h[t_num], flag_g[t_num];
     
     cudaMalloc((void**)&dist_g, N * N * sizeof(float));
     cudaMalloc((void**)&T_g, sizeof(float));
     cudaMalloc((void**)&r_g, t_num * sizeof(float));
     cudaMalloc((void**)&i_g, t_num * sizeof(int));
     cudaMalloc((void**)&k_g, t_num * sizeof(int));
     cudaMalloc((void**)&salesman_route_g, N * sizeof(int));
     cudaMalloc((void**)&flag_g, t_num * sizeof(int));
     
     // Beta is the temporary decay rate
     float beta = 0.95;
     float a = 1; 
     float f;
     
     while (T > 1){
         // Init parameters
         for(m = 0; m < t_num; m++){
             // pick first city to swap
             i_h[m] = rand() % N;
             // f defines how far the second city can be from the first
             f = exp(-a / T);
             j = 1 + rand() % (int)floor(1 + N * f);
             // pick second city to swap
             k_h[m] = (i_h[m] + j) % N;
             r_h[m] = rand() / 2147483647.0;
          }
          cudaMemcpy(i_g, i_h, t_num* sizeof(int), cudaMemcpyHostToDevice);
          cudaMemcpy(k_g, k_h, t_num* sizeof(int), cudaMemcpyHostToDevice);
          // What is order of operation for N*N*?
          cudaMemcpy(dist_g, dist, (N*N)* sizeof(float), cudaMemcpyHostToDevice);
          cudaMemcpy(salesman_route_g, salesman_route, N* sizeof(int), cudaMemcpyHostToDevice);
          cudaMemcpy(T_g, &T, sizeof(float), cudaMemcpyHostToDevice);
          cudaMemcpy(r_g, r_h, t_num* sizeof(float), cudaMemcpyHostToDevice);
          cudaMemcpy(flag_g, flag_h, t_num* sizeof(int), cudaMemcpyHostToDevice);
          
          tsp<<< 1, t_num, 0>>>(i_g, k_g, dist_g, salesman_route_g, T_g, r_g, flag_g);
          
          cudaMemcpy(flag_h, flag_g, t_num * sizeof(int), cudaMemcpyDeviceToHost);
          cudaMemcpy(salesman_route, salesman_route_g, N* sizeof(int), cudaMemcpyHostToDevice);
          
          /* 
          Here we check for a success
            The first proposal trip accepted becomes the new starting trip 
          
          for (i = 0; i < t_num; i++){
              if (flag_h[i] == 0){
                  continue;
              } else {
                  // Make all the new starting trips equal to the accepted winner
                  for (j = 0; j < N; j++){
                      salesman_route[j] = salesman_route[i];
                  }
                  //decrease temp
                  T -= T*beta;
                  printf("Current Temperature is %.6f", T)
                  printf("Best found trip so far\n")
                  for (j = 0; j < N; j++){
                     printf("%d", salesman_route[j])
                  }
              }
          }
          */
             
     // Do we free memory in each while loop?    
     cudaFree(i_g);
     cudaFree(k_g);
     cudaFree(dist_g);
     cudaFree(salesman_route_g);
     cudaFree(&T_g);
     cudaFree(r_g);
     cudaFree(flag_g);
     
     /*
     This is to check the flags
     for(m = 0; m < t_num; m++){
         printf("%d\n", flag_h[m]);
     }
     getchar();
     getchar();
     */
     }
     return 0;
}
             
         
         
         
         
