#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>

#define t_num 10


__global__ static void rd(int* seed)
{
	const int tid = threadIdx.x;
	int a = 16807, q = 127773, z = 2836, r=seed[tid];   //note that these parameters can't be adjusted, they're optimized, by some unknown math theory

	r = a*(r%q) - z*(r / q);
	if (r < 0)
		r += 2147483647;
	seed[tid] = r;                   //then the seed, or r is desired, unnormalized integer random number, range (0,2147483647)
}

int main()
{
	int r[10];   //we have to creat an array of random seed from host
	int i,length,j;
	for (i = 0; i<10; i++)
	{
		r[i] = rand();     //  generate seed
	}
	length = 10;
	int *r_g;

	cudaMalloc((void**)&r_g, length*sizeof(int));
	cudaMemcpy(r_g, r, sizeof(int)* length, cudaMemcpyHostToDevice);  //transport seeds

	for (i = 1; i < 3; i++)
	{
		rd << <1, t_num, 0 >> >(r_g);     //then just call the function no matter how many times you like, we can get diffrerent random numbers
		cudaMemcpy(r, r_g, length*sizeof(int), cudaMemcpyDeviceToHost);
		for (j = 1; j < 10; j++)
		{
			printf("int rand %d ", r[j]);
			printf("normalized rand %f\n", (float)r[j] / 2147483647);
		}
	}
	cudaFree(r_g);
	getchar();

	return 0;
}