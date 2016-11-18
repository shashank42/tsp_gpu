#include <unistd.h>
#include <stdio.h>

/* we need these includes for CUDA's random number stuff */
#include <curand.h>
#include <curand_kernel.h>



#define MAX 100

/* this GPU kernel function is used to initialize the random states */
__global__ void init(unsigned int seed, curandState_t* states) {

  /* we have to initialize the state */
  curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
              blockIdx.x, /* the sequence number should be different for each core (unless you want all
                             cores to get the same sequence of numbers for some reason - use thread id! */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &states[blockIdx.x]);
}

/* this GPU kernel takes an array of states, and an array of ints, and puts a random int into each */
__global__ void randoms(curandState_t* states, unsigned int* city_one, unsigned int* city_two,
                        unsigned int * N, float * T_start, float *T,
                        unsigned int *flag, volatile unsigned int *global_flag, float *r) {
  /* curand works like rand - except that it takes a state as a parameter */
  int xid = blockIdx.x * blockDim.x + threadIdx.x;
  float myrandf = curand_uniform(states+xid);
  myrandf *= ((N[0]-2) - 0+0.999999);
  myrandf += 0;
  unsigned int myrand = (unsigned int)truncf(myrandf);
  city_one[xid] = myrand;
  float swap_distance = exp(-sqrt(log(T_start[0] / T[0])));
  unsigned int j = (unsigned int)floor(1 + city_one[xid] * swap_distance); 
  city_two[xid] = (city_one[xid] + j) % (N[0]-1);
  
  flag[xid] = 0;
  global_flag[0] = 0;
  r[xid] = (float)curand_uniform(states+xid);
}

int main( ) {
  /* CUDA's random number library uses curandState_t to keep track of the seed value
     we will store a random state for every thread  */
  curandState_t* states;
  unsigned int N = 25, *N_g;
  float T_start = 5, *T_start_g;
  float T = T_start, *T_g;
  unsigned int *flag_g, *global_flag_g;
  float r[N], *r_g;
  /* allocate space on the GPU for the random states */
  cudaMalloc((void**) &states, N * sizeof(curandState_t));
  cudaMalloc((void**) &N_g, sizeof(unsigned int));
  cudaMalloc((void**) &T_g, sizeof(float));
  cudaMalloc((void**) &T_start_g, sizeof(float));
  cudaMalloc((void**) &flag_g, N * sizeof(float));
  cudaMalloc((void**) &global_flag_g, sizeof(float));
  cudaMalloc((void**) &r_g, N * sizeof(float));
  cudaMemcpy(N_g, &N, sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(T_g, &T, sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(T_start_g, &T_start, sizeof(float), cudaMemcpyHostToDevice);
  /* invoke the GPU to initialize all of the random states */
  init<<<N, 1>>>(time(0), states);

  /* allocate an array of unsigned ints on the CPU and GPU */
  unsigned int cpu_nums[N];
  unsigned int* gpu_nums;
  cudaMalloc((void**) &gpu_nums, N * sizeof(unsigned int));
  unsigned int cpu_nums2[N];
  unsigned int* gpu_nums2;
  cudaMalloc((void**) &gpu_nums2, N * sizeof(unsigned int));

  /* invoke the kernel to get some random numbers */
  randoms<<<N, 1>>>(states, gpu_nums,gpu_nums2, N_g, T_start_g, T_g, flag_g, global_flag_g, r_g);

  /* copy the random numbers back */
  cudaMemcpy(cpu_nums, gpu_nums, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu_nums2, gpu_nums2, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
  cudaMemcpy(r, r_g, N * sizeof(float), cudaMemcpyDeviceToHost);
  /* print them out */
  for (int i = 0; i < N; i++) {
    printf("%u and %u and %.06f \n", cpu_nums[i], cpu_nums2[i], r[i]);
  }

  /* free the memory we allocated for the states and numbers */
  cudaFree(states);
  cudaFree(gpu_nums);
  cudaFree(N_g);

  return 0;
}
