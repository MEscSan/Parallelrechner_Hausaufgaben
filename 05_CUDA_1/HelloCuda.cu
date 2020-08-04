#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void hello(char *a, int *b) 
{
  for (int i=0; i<7; ++i)
  {
    a[i] += b[i];
  }
}


int main(int argc, char* argv[])
{
  // Hello Array
  char a[7] = "Hello ";
  // Array with paddings (last one must be 0, so the string 
  // terminates correctly)
  int b[7] = {15, 10, 6, 0, -11, 1, 0};

  // pointers to device arrays
  char *ad;
  int *bd;

  // print "Hello "
  printf("%s", a);

  // size of memory to be copied
  size_t charArraySize = 7*sizeof(char);
  size_t intArraySize = 7*sizeof(int);

  // alloc memory on device and copy the data
  cudaMalloc( (void**)&ad, charArraySize ); 
  cudaMalloc( (void**)&bd, intArraySize ); 
  cudaMemcpy( ad, a, charArraySize, cudaMemcpyHostToDevice ); 
  cudaMemcpy( bd, b, intArraySize, cudaMemcpyHostToDevice ); 

  // start the thread on the device
  hello<<< 1, 1 >>>(ad, bd);

  // copy back the result
  cudaMemcpy( a, ad, charArraySize, cudaMemcpyDeviceToHost ); 
  // free the device memory
  cudaFree( ad );
  cudaFree( bd );

  // print the padded string "World!"
  printf("%s\n", a);
  return EXIT_SUCCESS;
}
