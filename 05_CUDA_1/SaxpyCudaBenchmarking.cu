#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__
void saxpy(int n, float a, float *x, float *y)
{
  printf("Hi from thread %d\n", threadIdx.x);
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

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  int numSMs;
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

  cudaEventRecord(start);
  saxpy<<<32, 64>>>(N, 2.0f, d_x, d_y);
  cudaEventRecord(stop);

  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmaxf(maxError, fabsf(y[i]-4.0f));
  printf("Max error: %f\n", maxError);

  printf("Elapsed Time: %f\n", milliseconds);

  // for Memory Throughput Benchmarking:
  printf("Effective Memory Throughput: %f GB/s\n", 4.0*N*3.0/milliseconds/1.0e6);
  int clockRate;
  int busWidth;
  cudaDeviceGetAttribute(&clockRate, cudaDevAttrMemoryClockRate, 0);
  cudaDeviceGetAttribute(&busWidth, cudaDevAttrGlobalMemoryBusWidth, 0);
  printf("Peak Memory Throughput: %f GB/s\n", 2.0*clockRate*(busWidth/8.0)/1.0e6);

  // for Computational Throughput Benchmarking:
  // (1* multiply + 1*add) times the amount of vector entries
  printf("Effective computational Throughput: %f GFLOP/s\n",(2*N/milliseconds)/1.0e6);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);
}
