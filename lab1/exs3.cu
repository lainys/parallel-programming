#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h> 

#include <iostream>
#include <fstream>

float check(int* a, int*b, int*c, int N, int k);
float cudaParallel(int *a, int *b, int *c, int N);

float checkThrust(int* a, int*b, int*c, int N, int k);
float cudaParallelThrust(int *a, int *b, int *c, int N);

struct sum_functor
{
	__host__  __device__
		float operator () (const  float & x, const  float & y) const {
		return  x + y;
	}
};

thrust::device_vector<int> multi_fast(thrust::device_vector < int > & X, thrust::device_vector < int > & Y)
{
	// Y <-  X + Y
	thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), sum_functor());
	return Y;
}

__global__ void kernel(int* a, int* b, int* c)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	c[idx] = a[idx] + b[idx];

}

template<typename T>
T init(T a, int N) {

	for (int i = 0; i < N; i++) {
		a[i] = 1;
	}
	return a;
}


int main()
{
	int k = 7;

	float *ans1 = new float[k];
	float *ans2 = new float[k];
	int *var_N = new int[k];

	for (int i = 0, N = 10; i < k; i++, N *= 10) {
		var_N[i] = N;

		int *a = new int[N];
		int *b = new int[N];
		int *c = new int[N];

		init(a, N);
		init(b, N);

		ans1[i] = check(a, b, c, N, 100);
		ans2[i] = check(a, b, c, N, 100);


		delete a, b, c;

		std::cout << i << std::endl;
	}

	std::ofstream out("text3.txt");

	for (int i = 0; i < k; i++) {
		out << var_N[i] << " ";
	}
	out << std::endl;

	for (int i = 0; i < k; i++) {
		out << ans1[i] << " ";
	}
	out << std::endl;

	for (int i = 0; i < k; i++) {
		out << ans2[i] << " ";
	}
	out << std::endl;

	delete ans1,ans2, var_N;
	return 0;
}

float check(int* a, int*b, int*c, int N, int k) {
	float res = 0;

	for (int i = 0; i < k; i++) {
		res += cudaParallel(a, b, c, N);
	}

	return res / (k * 1000.0f);
}


float checkThrust(int* a, int*b, int*c, int N, int k) {
	float res = 0;

	for (int i = 0; i < k; i++) {
		res += cudaParallelThrust(a, b, c, N);
	}

	return res / (k * 1000.0f);
}


float cudaParallel(int *a, int *b, int *c, int N) {

	int* dA = nullptr;
	int* dB = nullptr;
	int* dC = nullptr;

	cudaMalloc((void**)&dA, N * sizeof(int));
	cudaMalloc((void**)&dB, N * sizeof(int));
	cudaMalloc((void**)&dC, N * sizeof(int));

	cudaMemcpy(dA, a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dB, b, N * sizeof(int), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	float gpuTime = 0.0f;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	kernel << <dim3(N / 10, 1), dim3(10, 1) >> > (dA, dB, dC);

	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&gpuTime, start, stop);
	//printf("time spent executing by the GPU: %.5f ms\n", gpuTime);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaMemcpy(c, dC, N * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);


	return gpuTime;
}

float cudaParallelThrust(int *a, int *b, int *c, int N) {
	
	thrust::device_vector<int> at(N);
	thrust::device_vector<int> bt(N);

	for (int i = 0; i < N; i++) {
		at[i] = a[i];
		bt[i] = b[i];
	}

	cudaEvent_t start, stop;
	float gpuTime = 0.0f;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	multi_fast(at, bt);

	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&gpuTime, start, stop);
	//printf("time spent executing by the GPU: %.5f ms\n", gpuTime);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	for (int i = 0; i < N; i++) {
		c[i] = b[i];
	}

	return gpuTime;
}

