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

#include <curand.h>
#include <curand_kernel.h>


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
// Complex subtraction
static __device__ __host__ inline Complex ComplexSub(Complex a, Complex b)
{
    Complex `c;
    c.x = a.x - b.x;
    c.y = a.y - b.y;
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

static __global__ void ComputeDeltaH(Complex* delta_h, Complex* elastic,Complex* visc, Complex* randomforce, float k1, float k2, float f, float dt, int size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
        delta_h[i] = dt*(randomforce[i]+k2*(elastic[i]-visc[i])+k1*elastic[i] + f);
}

static __global__ void ComputeDeltaVisc(Complex* delta_u, Complex* elastic,Complex* visc,float k2, float dt, int size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
        delta_u[i] = dt*k2*(elastic[i]-visc[i]);
}




static __global__ void UpdateDisorder(Complex* fth, Complex* delta, float scale,  unsigned long seed, int size) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {

        curandState state;

        curand_init(seed, i, 0, &state);
        fth[i] = fth[i]*expf(-delta[i]/scale) + curand_normal(&state)*sqrtf(scale*(1-expf(-2*delta[i]/scale)));
    }
}



int main()
{
    const int nx = 256;
    const int ny = 256;
    const int numElements = nx*ny;
    const float alpha = 1.0;
    float PI = 4*atan(1.0);

    const float dt = 0.01;
    const float k1 = 1;
    const float k2= 0.5;
    const float f = 1;

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0,1.0);

    Complex* delta_int = (Complex*)malloc(sizeof(Complex) * numElements);
    Complex* delta_visc = (Complex*)malloc(sizeof(Complex) * numElements);
    Complex* interface = (Complex*)malloc(sizeof(Complex) * numElements);
    Complex* visc = (Complex*)malloc(sizeof(Complex) * numElements);
    Complex* force = (Complex*)malloc(sizeof(Complex) * numElements);
    Complex* vforce = (Complex*)malloc(sizeof(Complex) * numElements);
    Complex* rforce = (Complex*)malloc(sizeof(Complex) * numElements);
    float* kernel = (float*)malloc(sizeof(float) * numElements);
    for(int i = 0; i < numElements; i++)
    {
        int i_x = i % ny;
        int i_y = i / ny;
        kernel[i] = pow(4-2*cos(2*PI*fftfreq(i_x,nx))-2*cos(2*PI*fftfreq(i_y,ny)),alpha/2);
        delta_int[i].x = 0;
        delta_int[i].y = 0;
        delta_visc[i].x = 0;
        delta_visc[i].y = 0;
        interface[i].x = 0;
        interface[i].y = 0;
        visc[i].x = 0;
        visc[i].y = 0;
        force[i].x = 0;
        force[i].y = 0;
        vforce[i].x = 0;
        vforce[i].y = 0;
    }

    Complex* d_delta_int;
    checkCudaErrors(cudaMalloc((void **) &d_delta_int, numElements*sizeof(Complex))); 
    checkCudaErrors(cudaMemcpy(d_delta_int, delta_int, numElements*sizeof(Complex), cudaMemcpyHostToDevice));

    Complex* d_delta_visc;
    checkCudaErrors(cudaMalloc((void **) &d_delta_visc, numElements*sizeof(Complex))); 
    checkCudaErrors(cudaMemcpy(d_delta_visc, delta_int, numElements*sizeof(Complex), cudaMemcpyHostToDevice));

    Complex* d_interface;
    checkCudaErrors(cudaMalloc((void **) &d_interface, numElements*sizeof(Complex))); 
    checkCudaErrors(cudaMemcpy(d_interface, interface, numElements*sizeof(Complex), cudaMemcpyHostToDevice));

    Complex* d_visc;
    checkCudaErrors(cudaMalloc((void **) &d_visc, numElements*sizeof(Complex))); 
    checkCudaErrors(cudaMemcpy(d_visc, visc, numElements*sizeof(Complex), cudaMemcpyHostToDevice));

    Complex* d_force;
    checkCudaErrors(cudaMalloc((void **) &d_force, numElements*sizeof(Complex))); 
    checkCudaErrors(cudaMemcpy(d_force, force, numElements*sizeof(Complex), cudaMemcpyHostToDevice));

    Complex* d_vforce;
    checkCudaErrors(cudaMalloc((void **) &d_vforce, numElements*sizeof(Complex))); 
    checkCudaErrors(cudaMemcpy(d_vforce, vforce, numElements*sizeof(Complex), cudaMemcpyHostToDevice));

    Complex* d_rforce;
    checkCudaErrors(cudaMalloc((void **) &d_rforce, numElements*sizeof(Complex))); 
    checkCudaErrors(cudaMemcpy(d_rforce, rforce, numElements*sizeof(Complex), cudaMemcpyHostToDevice));

    float* d_kernel;
    checkCudaErrors(cudaMalloc((void **) &d_kernel, numElements*sizeof(float))); 
    checkCudaErrors(cudaMemcpy(d_kernel, kernel, numElements*sizeof(float), cudaMemcpyHostToDevice));




    dim3 numBlocks(nx/BLOCK_SIZE_X , ny/BLOCK_SIZE_Y );
    dim3 threadsPerBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);


    cufftHandle plan;
    cufftResult plan_res = cufftPlan2d(&plan, nx, ny, CUFFT_C2C);
    //cout << plan_res << endl;
    

    for(int t = 0; t < 100; t++)
    {
        
        //we compute the laplacian of the interface
        cufftExecC2C(plan, (Complex*)d_interface, (cufftComplex *)d_force, CUFFT_FORWARD);
        ComplexPointwiseScale<<<numBlocks, threadsPerBlock>>>(d_force, d_kernel, numElements);

        ComputeDeltaH<<<numBlocks, threadsPerBlock>>>(delta_h, d_force, d_visc, rforce, k1, k2, f, dt,numElements);
        ComputeDeltaVisc<<<numBlocks, threadsPerBlock>>>(delta_visc, d_force, d_visc,k2,dt,numElements);
    }

}


