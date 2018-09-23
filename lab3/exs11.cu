
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <fstream>
#define BLOCK_SIZE 16

float checkGPU(int *a, int N, int k);
float checkCPU(int *a, int N, int k);
float cudaParallel(int *a, int N);
float minCPU(int *a, int N);

__global__ void min(int * inData, int N)
{

	int tid = threadIdx.x;

	int k = blockIdx.x * blockDim.x + threadIdx.x;

	int a1 = k;
	int a2 = k + 1;

	while (a2 < N) {
		inData[a1] = (inData[a1] < inData[a2] ? inData[a1] : inData[a2]);

		a1 *= 2;
		a2 *= 2;

		if (a1 >= N) {
			break;
		}
		__syncthreads();
	}

}

void init(int* a, int N) {
	for (int i = 0; i < N; i++) {
		a[i] = 100 - i + 1;
	}
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

		init(a, N);


		ans1[i] = checkCPU(a, N, 100);
		ans2[i] = checkGPU(a, N, 100);


		std::cout << i << std::endl;

		delete a;
	}

	std::ofstream out("text1.txt", 'w');

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

	out.close();

	delete ans1, ans2, var_N;
	
	return 0;
}

float checkGPU(int* a, int N, int k) {
	float res = 0;

	for (int i = 0; i < k; i++) {
		res += cudaParallel(a, N);
	}

	return res / (k * 1.0f);
}

float checkCPU(int* a, int N, int k) {
	float res = 0;

	for (int i = 0; i < k; i++) {
		res += minCPU(a, N);
	}

	return res / (k * 1.0f);
}


float cudaParallel(int *a, int N) {

	int* dev_a;


	cudaMalloc((void**)&dev_a, N * sizeof(int));
	cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);


	//----
	cudaEvent_t start, stop;
	float gpuTime = 0.0f;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
	//----
	min << <1, N >> > (dev_a, N);

	//----
	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuTime, start, stop);

	//printf("N = %d, time spent executing by the GPU: %.5f ms\n",N, gpuTime);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	//----

	cudaMemcpy(a, dev_a, N * sizeof(int), cudaMemcpyDeviceToHost);


	cudaFree(dev_a);


	return gpuTime;
}

float minCPU(int* a, int N) {

	int min = a[0];
	//----
	cudaEvent_t start, stop;
	float gpuTime = 0.0f;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
	//----
	for (int i = 1; i < N; i++) {
		if (a[i] < min) {
			min = a[i];
		}
	}

	//----
	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuTime, start, stop);

	//printf("N = %d, time spent executing by the GPU: %.5f ms\n",N, gpuTime);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	//----

	a[0] = min;
	return gpuTime;
}