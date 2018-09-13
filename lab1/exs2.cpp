#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>

__global__ void kernel(int* a, int* b, int* c)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	c[idx] = a[idx] + b[idx];
	
}

template<typename T>
T init(T a,int N) {

	for (int i = 0; i < N; i++) {
		a[i] = 1;
	}
	return a;
}

int main(int argc, char *argv[])
{
	int N = 10;

	int* a = (int*) malloc(N * sizeof(int));
	int* b = (int*) malloc(N * sizeof(int));
	int* c = (int*) malloc(N * sizeof(int));

	int* dA = nullptr;
	int* dB = nullptr;
	int* dC = nullptr;

	a = init(a, N);
	b = init(b, N);

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

	kernel <<<dim3(N/10,1), dim3(10,1)>>> (dA,dB,dC);

	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&gpuTime, start, stop);
	printf("time spent executing by the GPU: %.5f ms\n", gpuTime);
	
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaMemcpy(c, dC, N * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);


	for (int i = 0; i < N; i++) {
		printf("%d\n", c[i]);
	}

	free(a);
	free(b);
	free(c);

	return 0;
}
