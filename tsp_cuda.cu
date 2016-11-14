#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>

#define t_num 256
#define N 100

/* BEGIN KERNEL
Input:
- city_one: A vector of cities to swap for the first swap choice
- city_two: A vector of cities to swap for the second swap choice
- dist: The distance matrix of each city
- salesman_route: The route the salesman will travel
- T: The current temperature
- r: The random number to compare against for S.A.
*/
__global__ static void tsp(int* city_one, int* city_two, float *dist, int *salesman_route,
                           float *T, float *r, int *flag){
    
    const int tid = threadIdx.x;
    float delta, p, b = 1;
    
    // first city to swap
    int salesman_route_city_one = salesman_route[city_one[tid]];
    int salesman_route_city_one_minus_mod = salesman_route[(city_one[tid] - 1 + N) % N];
    int salesman_route_city_one_plus_mod  = salesman_route[(city_one[tid] + 1) % N];
    
    // second city to swap
    int salesman_route_city_two = salesman_route[city_two[tid]];
    int salesman_route_city_two_plus_mod  = salesman_route[city_two[tid] + 1 % N];
    int salesman_route_city_two_minus_mod = salesman_route[(city_two[tid] - 1 + N) % N];
    
    // we should return this so we know the minimum route? -S.
    delta = dist[salesman_route_city_one_minus_mod * N + salesman_route_city_two] +
            dist[salesman_route_city_two * N + salesman_route_city_one_plus_mod]  +
            dist[salesman_route_city_one_minus_mod * N + salesman_route_city_one] +
            dist[salesman_route_city_one * N + salesman_route_city_two_plus_mod]  -
            dist[salesman_route_city_one_minus_mod * N + salesman_route_city_one] - 
            dist[salesman_route_city_one * N + salesman_route_city_one_plus_mod] -
            dist[salesman_route_city_two_minus_mod * N + salesman_route_city_two] - 
            dist[salesman_route_city_two * N + salesman_route_city_two_plus_mod];
    p = exp(-delta * b / T[0]);
    if (p > r[tid])
      flag[tid] = 1;
    else
      flag[tid] = 0;
 }
 
 /* BEGIN KERNEL
Input:
- city_one: [size(threads)]
    - An integer vector of cities to swap for the first swap choice
- city_two: [size(threads)]
    - An integer vector of cities to swap for the second swap choice
- dist: [size(N * N)] 
    - A floating point distance matrix of each city
- salesman_route: [size(threads * N)]
    - An integer matrix of the route the salesman will travel
- original_loss: [size(1)]
    - A floating point giving the original trips loss function
- new_loss: [size(threads)]
    - A floating point vector of the proposed trips loss function
- T: [size(1)]
    - A floating point of the current temperature
- r: [size(threads)]
    - A floating point the random number to compare against for S.A.
*/
 __global__ static void tspLoss(int* city_one, int* city_two, float *dist,
                                int *salesman_route, float *original_loss, float *new_loss,
                                float *T, float *r, int* flag){
    
    const int tid = threadIdx.x;
    float delta, p, b = 1;
    float sum = 0;
    // make the proposal route
    int proposal_route[N];
    for (int i = 0; i < N; i++)
        proposal_route[i] = salesman_route[i];
    
    // Do the switch    
    proposal_route[city_one[tid]] = salesman_route[city_two[tid]];
    proposal_route[city_two[tid]] = salesman_route[city_one[tid]];
    
    // evaluate new route's loss function
    for (int i = 0; i < N - 1; i++)
         sum += dist[proposal_route[i] * N + proposal_route[i + 1]];
    
    
    // Acceptance / Rejection step
    if (sum < original_loss[0]){
        flag[tid] = 1;
        new_loss[tid] = sum;
    } else {
        delta = original_loss[0] - sum;
        p = exp(-(delta * b / T[0]));
        if (p > r[tid]){
            flag[tid] = 1;
            new_loss[tid] = sum;
        } else {
            flag[tid] = 0;
        }
    }
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
     
     int salesman_route[N];
     
     // just make one inital guess route, a simple linear path
     for (int i = 0; i < N; i++)
         salesman_route[i] = i;
         
     /* the order of the salesman problem
     int salesman_route[N * t_num];
     int init_route[N];
     
     //make an initial route
     for (int i = 0; i < N; i++){
         init_route[i] = rand() % (int)floor(1 + N);
     }
     // fill the routes for the kernel with our initial guess
     for (int i = 0; i < t_num; i++){
       for (int j = 0; j < N; j++){
           salesman_route[i * N + j] = init_route[j];
       }
     }
     */
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
     
     // Calculate the original loss
     float original_loss = 0;
     for (int i = 0; i < N - 1; i++){
         original_loss += dist[salesman_route[i] * N + salesman_route[i+1]];
     }
     
     float dist_g[N * N], T = 5, T_g[1], r_h[t_num], r_g[t_num];
     
     /*
     Defining device variables:
     city_swap_one_h/g: 
       - Host/Device memory for city one
     city_swap_two_h/g:
       - Host/Device memory for city two
     flag_h/g:
       - Host/Device memory for flag of accepted step
     salesman_route_g:
       - Device memory for the salesmans route
     r_g:  
       - Device memory for the random number when deciding acceptance
     flag_h/g:
       - host/device memory for acceptance vector
     original_loss_g: 
       - The device memory for the current loss function
     new_loss_h/g: 
       - The host/device memory for the proposal loss function
     */
     int city_swap_one_h[t_num], city_swap_one_g[t_num],
         city_swap_two_h[t_num], city_swap_two_g[t_num],
         salesman_route_g[N],
         flag_h[t_num], flag_g[t_num];
         
     float original_loss_g[1];
     float new_loss_h[t_num] = {0.0}, new_loss_g[t_num];   
     
     cudaMalloc((void**)&city_swap_one_g, t_num * sizeof(int));
     cudaMalloc((void**)&city_swap_two_g, t_num * sizeof(int));
     cudaMalloc((void**)&dist_g, N * N * sizeof(float));
     cudaMalloc((void**)&salesman_route_g, N * t_num * sizeof(int));
     cudaMalloc((void**)&original_loss_g, sizeof(float));
     cudaMalloc((void**)&new_loss_g, t_num * sizeof(float));
     cudaMalloc((void**)&T_g, sizeof(float));
     cudaMalloc((void**)&r_g, t_num * sizeof(float));
     cudaMalloc((void**)&flag_g, t_num * sizeof(int));
     // Beta is the temporary decay rate
     float beta = 0.95;
     float a = 1; 
     float f;
     
     while (T > 1){
         // Init parameters
         printf("Current Temperature is: %.6f:", T);
         for(m = 0; m < t_num; m++){
             // pick first city to swap
             city_swap_one_h[m] = rand() % N;
             // f defines how far the second city can be from the first
             f = exp(-a / T);
             j = 1 + rand() % (int)floor(1 + N * f);
             // pick second city to swap
             city_swap_two_h[m] = (city_swap_one_h[m] + j) % N;
             r_h[m] = rand() / 2147483647.0;
             
             //set our flags and new loss to 0
             flag_h[m] = 0;
             new_loss_h[m] = 0;
          }
          cudaMemcpy(city_swap_one_g, city_swap_one_h, t_num* sizeof(int), cudaMemcpyHostToDevice);
          cudaMemcpy(city_swap_two_g, city_swap_two_h, t_num* sizeof(int), cudaMemcpyHostToDevice);
          cudaMemcpy(dist_g, dist, (N*N)* sizeof(float), cudaMemcpyHostToDevice);
          cudaMemcpy(salesman_route_g, salesman_route, N * sizeof(int), cudaMemcpyHostToDevice);
          cudaMemcpy(T_g, &T, sizeof(float), cudaMemcpyHostToDevice);
          cudaMemcpy(r_g, r_h, t_num * sizeof(float), cudaMemcpyHostToDevice);
          cudaMemcpy(flag_g, flag_h, t_num* sizeof(int), cudaMemcpyHostToDevice);
          cudaMemcpy(original_loss_g, &original_loss, sizeof(float), cudaMemcpyHostToDevice);
          cudaMemcpy(new_loss_g, new_loss_h, t_num * sizeof(float), cudaMemcpyHostToDevice);
          
          //tsp<<< 1, t_num, 0>>>(city_swap_one_g, city_swap_two_g, dist_g, salesman_route_g, T_g, r_g, flag_g);
          tspLoss<<<1, t_num, 0>>>(city_swap_one_g, city_swap_two_g, dist_g, salesman_route_g,
                                   original_loss_g, new_loss_g, T_g, r_g, flag_g);
                    
          cudaMemcpy(flag_h, flag_g, t_num * sizeof(int), cudaMemcpyDeviceToHost);
          // In tspLoss, we don't actually need the newest route to be copied over
          //cudaMemcpy(salesman_route, salesman_route_g, N* sizeof(int), cudaMemcpyHostToDevice);
          cudaMemcpy(new_loss_h, new_loss_g, t_num * sizeof(float), cudaMemcpyHostToDevice);
          
          /* 
          Here we check for a success
            The first proposal trip accepted becomes the new starting trip 
          */
          for (i = 0; i < t_num; i++){
              if (flag_h[i] == 0){
                  continue;
              } else {
                  // switch the two cities that led to an accepted proposal
                  int tmp = salesman_route[city_swap_one_h[i]];
                  salesman_route[city_swap_one_h[i]] = salesman_route[city_swap_two_h[i]];
                  salesman_route[city_swap_two_h[i]] = tmp;
                  
                  // set old loss function to new
                  original_loss = new_loss_h[i];
                  //decrease temp
                  T -= T*beta;
                  printf("Current Temperature is %.6f", T);
                  printf("Best found trip so far\n");
                  for (j = 0; j < N; j++){
                     printf("%d", salesman_route[j]);
                  }
                  break;
              }
          }
          
             
     
     /*
     This is to check the flags
     for(m = 0; m < t_num; m++){
         printf("%d\n", flag_h[m]);
     }
     getchar();
     getchar();
     */
     }
         
     cudaFree(city_swap_one_g);
     cudaFree(city_swap_two_g);
     cudaFree(dist_g);
     cudaFree(salesman_route_g);
     cudaFree(&T_g);
     cudaFree(r_g);
     cudaFree(flag_g);
     cudaFree(&new_loss_g);
     cudaFree(original_loss_g);
     return 0;
}
             
         
         
         
         
