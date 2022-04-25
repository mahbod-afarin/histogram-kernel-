/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

// Define your kernels in this file you may use more than one kernel if you
// need to

// INSERT KERNEL(S) HERE

__global__ void histogram_kernel(unsigned int *buffer, unsigned int size, unsigned int histogram_size, unsigned int *output) 
{

    extern __shared__ int histogram_prv[];
    unsigned int m = (histogram_size - 1 / blockDim.x) + 1; 
  
    if (threadIdx.x < histogram_size) 
    {
      for (unsigned int j = 0; j < m && (threadIdx.x + (j)*blockDim.x) < histogram_size; j++)
        histogram_prv[threadIdx.x + j * blockDim.x] = 0;
    }
    __syncthreads();
  
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int str = blockDim.x * gridDim.x;
    while (i < size) 
    {
      atomicAdd(&(histogram_prv[buffer[i]]), 1);
      i += str;
    }
    __syncthreads();
  
    if (threadIdx.x < histogram_size) 
    {
      for (unsigned int j = 0; j <= m && (threadIdx.x + (j)*blockDim.x) < histogram_size; j++)
        atomicAdd(&(output[threadIdx.x + j * blockDim.x]),
        histogram_prv[threadIdx.x + j * blockDim.x]);
    }
  }
  
  /******************************************************************************
  Setup and invoke your kernel(s) in this function. You may also allocate more
  GPU memory if you need to
  *******************************************************************************/
  void histogram(unsigned int *input, unsigned int *bins,
                 unsigned int num_elements, unsigned int num_bins) {
  
    // INSERT CODE HERE
  
    int BLOCK_SIZE = 512;
  
    dim3 dim_grid(((num_elements - 1) / BLOCK_SIZE) + 1, 1, 1);
    dim3 dim_block(BLOCK_SIZE, 1, 1);
  
    int histogram_prv_size = num_bins * (sizeof(int));
    histogram_kernel<<<dim_grid, dim_block, histogram_prv_size>>>(input, num_elements, num_bins, bins);

  }
  