#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>

#define t_num 256
#define N 100

// BEGIN KERNEL
__global__ static void try(int* i, int* k, float *dist, int *odr, float *T, float *r, int *flag){
    
    const int tid = threadIdx.x;
    float delta, p, b = 1;
    
    int odr_k = odr[k[tid]];
    int odr_i = odr[i[tid]];
    int odr_kplus_mod  = odr[k[tid] + 1 % N];
    int odr_kminus_mod = odr[(k[tid] - 1 + N) % N];
    int odr_iminus_mod = odr[(i[tid] - 1 + N) % N];
    int odr_iplus_mod  = odr[(i[tid] + 1) % N];
    
    delta = dist[odr_iminus_mod * N + odr_k] + dist[odr_k * N + odr_iplus_mod] +
            dist[odr_kminus_mod * N + odr_i] + dist[odr_i * N + odr_kplus_mod] -
            dist[odr_iminus_mod * N + odr_i] - dist[odr_i * N + odr_iplus_mod] -
            dist[odr_kminus_mod * N + odr_k] - dist[odr_k * N + odr_kplus_mod];
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
     struct location {
         int x;
         int y;
     };
     
     struct location lct[N];
     
     // the order of the salesman problem
     int odr[N];
     
     // initialize the location and sequence
     for(i = 0; i < N; i++){
         lct[i].x = rand() % 1000;
         lct[i].y = rand() % 1000;
     }
     
     // distance
     float dist[N * N];
     
     for(i = 0; i < N; i++){
         for (j = 0; j < N; j++){
             // Calculate the euclidian distance between each city
             // use pos() here instead?
             dist[i * N + j] = (lct[i].x - lct[j].x) * (lct[i].x - lct[j].x) +
                               (lct[j].y - lct[j].y) * (lct[i].y - lct[j].y);
         }
     }
     
     float dist_g[N * N], T = 1000, T_g[1], r_h[t_num], r_g[t_num];
     
     /*
     Defining device variables
     r_g is the random number for deciding acceptance
     flag is the acceptance vector
     */
     int i_h[t_num], i_g[t_num],    k_h[t_num], k_g[t_num],
       odr_g[N],  flag_h[t_num], flag_g[t_num];
     
     cudaMalloc((void**)&dist_g, N * N * sizeof(float));
     cudaMalloc((void**)&T_g, sizeof(float));
     cudaMalloc((void**)&r_g, t_num * sizeof(float));
     cudaMalloc((void**)&i_g, t_num * sizeof(int));
     cudaMalloc((void**)&k_g, t_num * sizeof(int));
     cudaMalloc((void**)&odr_g, N * sizeof(int));
     cudaMalloc((void**)&flag_g, t_num * sizeof(int));
     
     // Beta is the temorary decay rate
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
          cudaMemcpy(odr_g, odr, N* sizeof(int), cudaMemcpyHostToDevice);
          cudaMemcpy(T_g, &T, sizeof(float), cudaMemcpyHostToDevice);
          cudaMemcpy(r_g, r_h, t_num* sizeof(float), cudaMemcpyHostToDevice);
          cudaMemcpy(flag_g, flag_h, t_num* sizeof(int), cudaMemcpyHostToDevice);
          
          try<<< 1, t_num, 0>>>(i_g, k_g, dist_g, odr_g, T_g, r_g, flag_g);
          
          cudaMemcpy(flag_h, flag_h, t_num * sizeof(int), cudaMemcpyDeviceToHost);
          T = 0;
     }
     cudaFree(i_g);
     cudaFree(k_g);
     cudaFree(dist_g);
     cudaFree(odr_g);
     cudaFree(&T_g);
     cudaFree(r_g);
     cudaFree(flag_g);
     
     for(m = 0; m < t_num; m++){
         printf("%d\n", flag_h[m]);
     }
     getchar();
     getchar();
     return 0;
}
             
         
         
         
         
