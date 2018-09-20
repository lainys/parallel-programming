
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <fstream>

float cudaParallel(int *a, int *b, int *c, int N, bool flag);
float check(int* a, int*b, int*c, int N, bool flag, int k = 10);

__global__ void addKernel(int *a, int *b, int *c)
{
	int i = threadIdx.x;

	c[i] = a[i] * b[i];
}

__global__ void addKernel2(int *a, int *b, int *c)
{
	int i = threadIdx.x;

	if (i == 4) {
		i++;
	}
	else if (i == 5) {
		i--;
	}

	c[i] = a[i] * b[i];
}

void init(int* a, int N) {
	for (int i = 0; i < N; i++) {
		a[i] = i + 1;
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
		int *b = new int[N];
		int *c = new int[N];

		init(a, N);
		init(b, N);

		ans1[i] = check(a, b, c, N, true, 100);
		ans2[i] = check(a, b, c, N, false, 100);

		delete a, b, c;
	}

	std::ofstream out("text.txt");

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
	
	delete ans1, ans2,var_N;
	return 0;
}

float check(int* a, int*b, int*c, int N, bool flag, int k) {
	float res = 0;

	for (int i = 0; i < k; i++) {
		res += cudaParallel(a, b, c, N, flag);
	}

	return res / (k * 1.0f);
}


float cudaParallel(int *a, int *b, int *c, int N, bool flag) {
	int* dev_a;
	int* dev_b;
	int* dev_c;

	cudaMalloc((void**)&dev_a, N * sizeof(int));
	cudaMalloc((void**)&dev_b, N * sizeof(int));
	cudaMalloc((void**)&dev_c, N * sizeof(int));


	cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

	//----
	cudaEvent_t start, stop;
	float gpuTime = 0.0f;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
	//----

	if (flag) {
		addKernel << <1, N >> > (dev_a, dev_b, dev_c);
	}
	else {
		addKernel2 << <1, N >> > (dev_a, dev_b, dev_c);

	}

	//----
	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuTime, start, stop);

	//printf("N = %d, time spent executing by the GPU: %.5f ms\n",N, gpuTime);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	//----

	cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return gpuTime;
}
