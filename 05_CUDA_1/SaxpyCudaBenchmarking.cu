#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__
void saxpy(int n, float a, float *x, float *y)
{
	  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) 
	  {
	    y[i] = a * x[i] + y[i];
	    __syncthreads();
	  }
}

int main(void)
{
	int N = 1<<25;
	printf("%d\n", N);
	float *x, *y, *d_x, *d_y;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	  
	x = (float*)malloc(N*sizeof(float));
	y = (float*)malloc(N*sizeof(float));
	cudaMalloc(&d_x, N*sizeof(float)); 
	cudaMalloc(&d_y, N*sizeof(float));

	for (int n = 0; n < N; n++) {
	  x[n] = 1.0f;
	  y[n] = 2.0f;
	}

	cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

	int numSMs;
	cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);


	printf("Blocks	Threads	Max error	Elapsed time 	Eff. Memory Throughput	"
			 "Peak Memory Throughput	Eff. Computational Throughput\n");
for(int j =1; j < 10; j++){
	for(int i = 1; i < 10; i++){


		int numThreads = i*32, numBlocks = j *32; 	  
		printf("%d	", numBlocks);
                printf("%d	", numThreads);
   		cudaEventRecord(start);
		saxpy<<<numBlocks, numThreads>>>(N, 2.0f, d_x, d_y);
		cudaEventRecord(stop);

		cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);

		float maxError = 0.0f;
		for (int k = 0; k < N; k++)
		  maxError = fmaxf(maxError, fabsf(y[k]-4.0f));
		printf("%f	", maxError);

		printf("%f	", milliseconds);

		// for Memory Throughput Benchmarking:
		printf("%f GB/s		", 4.0*N*3.0/milliseconds/1.0e6);
		int clockRate;
		int busWidth;
		cudaDeviceGetAttribute(&clockRate, cudaDevAttrMemoryClockRate, 0);
		cudaDeviceGetAttribute(&busWidth, cudaDevAttrGlobalMemoryBusWidth, 0);
		printf("%f GB/s		", 2.0*clockRate*(busWidth/8.0)/1.0e6);

		// for Computational Throughput Benchmarking:
		// (1* multiply + 1*add) times the amount of vector entries
		printf("%f GFLOP/s	\n",(2*N/milliseconds)/1.0e6);
		}
	}
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(d_x);
	cudaFree(d_y);
	free(x);
	free(y);
}
