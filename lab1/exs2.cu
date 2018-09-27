#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <iostream>
#include <fstream>

float check(int* a, int*b, int*c, int N, int k);
float cudaParallel(int *a, int *b, int *c, int N);

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
	int *var_N = new int[k];

	for (int i = 0, N = 10; i < k; i++, N *= 10) {
		var_N[i] = N;

		int *a = new int[N];
		int *b = new int[N];
		int *c = new int[N];

		init(a, N);
		init(b, N);

		ans1[i] = check(a, b, c, N, 100);


		delete a, b, c;

		std::cout << i << std::endl;
	}

	std::ofstream out("text2.txt");

	for (int i = 0; i < k; i++) {
		out << var_N[i] << " ";
	}
	out << std::endl;

	for (int i = 0; i < k; i++) {
		out << ans1[i] << " ";
	}
	out << std::endl;

	delete ans1, var_N;
	return 0;
}

float check(int* a, int*b, int*c, int N, int k) {
	float res = 0;

	for (int i = 0; i < k; i++) {
		res += cudaParallel(a, b, c, N);
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
