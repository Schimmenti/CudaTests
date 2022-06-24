#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_functions.h>
#include <helper_cuda.h>

#include <ctime>
#include <time.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <cufft.h>
#include <fstream>



#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32


typedef float2 Complex;
using namespace std;

//converts matrix (i,j) with 'width' columns to a single index 
static int __host__ __device__ getIndex(const int i, const int j, const int width)
{
    return i*width + j;
}

static int __host__ __device__ fftfreq(const int i, const int N)
{
    if (i < N / 2)
    {
        return i;
    }
    else
    {
        return i - N;
    }
}

// Complex addition
static __device__ __host__ inline Complex ComplexAdd(Complex a, Complex b)
{
    Complex c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}
static __device__ __host__ inline Complex ComplexScale(Complex a, float s)
{
    Complex c;
    c.x = s * a.x;
    c.y = s * a.y;
    return c;
}
// Complex multiplication
static __device__ __host__ inline Complex ComplexMul(Complex a, Complex b)
{
    Complex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}
// Complex pointwise multiplication
static __global__ void ComplexPointwiseScale(Complex* a, const float* b, int size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
        a[i] = ComplexScale(a[i], b[i]);
}
// Complex pointwise multiplication
static __global__ void ComplexPointwiseMulAndScale(Complex* a, const Complex* b, int size, float scale)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
        a[i] = ComplexScale(ComplexMul(a[i], b[i]), scale);
}



int main()
{
    const int nx = 256;
    const int ny = 256;
    const int numElements = nx*ny;
    const float alpha = 1.0;
    float PI = 4*atan(1.0);

    Complex* interface = (Complex*)malloc(sizeof(Complex) * numElements);
    Complex* force = (Complex*)malloc(sizeof(Complex) * numElements);
    float* kernel = (float*)malloc(sizeof(float) * numElements);
    for(int i = 0; i < numElements; i++)
    {
        int i_x = i % ny;
        int i_y = i / ny;
        kernel[i] = pow(4-2*cos(2*PI*fftfreq(i_x,nx))-2*cos(2*PI*fftfreq(i_y,ny)),alpha/2);
        interface[i].x = 0;
        interface[i].y = 0;
        force[i].x = 0;
        force[i].y = 0;
    }

    Complex* d_interface;
    checkCudaErrors(cudaMalloc((void **) &d_interface, numElements*sizeof(Complex))); 
    checkCudaErrors(cudaMemcpy(d_interface, interface, numElements*sizeof(Complex), cudaMemcpyHostToDevice));

    Complex* d_force;
    checkCudaErrors(cudaMalloc((void **) &d_force, numElements*sizeof(Complex))); 
    checkCudaErrors(cudaMemcpy(d_force, force, numElements*sizeof(Complex), cudaMemcpyHostToDevice));

    float* d_kernel;
    checkCudaErrors(cudaMalloc((void **) &d_kernel, numElements*sizeof(float))); 
    checkCudaErrors(cudaMemcpy(d_kernel, kernel, numElements*sizeof(float), cudaMemcpyHostToDevice));

    dim3 numBlocks(nx/BLOCK_SIZE_X , ny/BLOCK_SIZE_Y );
    dim3 threadsPerBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);


    cufftHandle plan;
    cufftPlan2d(&plan, nx, ny, CUFFT_C2C);

    

    for(int t = 0; t < 100; t++)
    {
        cufftExecC2C(plan, (Complex*)d_interface, (cufftComplex *)d_force, CUFFT_FORWARD);
        ComplexPointwiseScale<<<numBlocks, threadsPerBlock>>>(d_force, d_kernel, numElements);
    }

}


