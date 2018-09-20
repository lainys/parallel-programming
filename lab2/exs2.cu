
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <fstream>

float cudaParallel(int *a, int *b, int *c, int N, bool flag, int n_stream);
float check(int* a, int*b, int*c, int N, bool flag, int stream, int k);

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
	int n_stream = 8;

	float *ans1 = new float[k*n_stream];
	float *ans2 = new float[k*n_stream];
	int *var_N = new int[k];

	for (int i = 0, N = 10; i < k; i++, N *= 10) {
		var_N[i] = N;

		int *a = new int[N];
		int *b = new int[N];
		int *c = new int[N];

		init(a, N);
		init(b, N);

		for (int j = 0; j < n_stream; j++) {

			ans1[j*k + i] = check(a, b, c, N, true, j, 100);
			ans2[j*k + i] = check(a, b, c, N, false, j, 100);
		}

		std::cout << i << std::endl;

		delete a, b, c;
	}

	std::ofstream out("text2.txt", 'w');

	out << n_stream<<std::endl;
	for (int i = 0; i < k; i++) {
		out << var_N[i] << " ";
	}
	out << std::endl;


	for (int i = 0; i < k*n_stream; i++) {
		out << ans1[i] << " ";
	}
	out << std::endl;

	for (int i = 0; i < k*n_stream; i++) {
		out << ans2[i] << " ";
	}
	out << std::endl;

	out.close();

	delete ans1, ans2, var_N;

	return 0;
}

float check(int* a, int*b, int*c, int N, bool flag, int stream, int k) {
	float res = 0;

	for (int i = 0; i < k; i++) {
		res += cudaParallel(a, b, c, N, flag, stream);
	}

	return res / (k * 1.0f);
}


float cudaParallel(int *a, int *b, int *c, int N, bool flag, int n_stream) {
	cudaStream_t* stream = new cudaStream_t[n_stream];
	int *addr = new int[n_stream];

	//address
	for (int i = 0; i < n_stream; i++) {
		addr[i] = N / n_stream + (N%n_stream - i > 0 ? 1 : 0);
	}

	// create
	for (int i = 0; i < n_stream; i++) {
		cudaStreamCreate(&stream[i]);
	}


	int* dev_a;
	int* dev_b;
	int* dev_c;

	// init memory
	cudaMalloc((void**)&dev_a, N * sizeof(int));
	cudaMalloc((void**)&dev_b, N * sizeof(int));
	cudaMalloc((void**)&dev_c, N * sizeof(int));

	// copy to device
	int offset = 0;
	for (int i = 0; i < n_stream; i++) {

		cudaMemcpyAsync(dev_a + offset, a + offset, addr[i], cudaMemcpyHostToDevice, stream[i]);
		cudaMemcpyAsync(dev_b + offset, b + offset, addr[i], cudaMemcpyHostToDevice, stream[i]);

		offset += addr[i];
	}

	//----
	cudaEvent_t start, stop;
	float gpuTime = 0.0f;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
	//----

	if (flag) {
		offset = 0;
		for (int i = 0; i < n_stream; i++) {

			addKernel << <1, addr[i], 0, stream[i] >> > (dev_a + offset, dev_b + offset, dev_c + offset);

			offset += addr[i];
		}
	}
	else {
		offset = 0;
		for (int i = 0; i < n_stream; i++) {

			addKernel2 << <1, addr[i], 0, stream[i] >> > (dev_a + offset, dev_b + offset, dev_c + offset);

			offset += addr[i];
		}

	}

	cudaDeviceSynchronize();
	//----
	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuTime, start, stop);

	//printf("N = %d, time spent executing by the GPU: %.5f ms\n",N, gpuTime);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	//----

	// copy to host
	offset = 0;
	for (int i = 0; i < n_stream; i++) {

		cudaMemcpyAsync(dev_c + offset, c + offset, addr[i], cudaMemcpyDeviceToHost, stream[i]);

		offset += addr[i];
	}

	cudaDeviceSynchronize();

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	//destroy streams
	for (int i = 0; i < n_stream; i++) {
		cudaStreamDestroy(stream[i]);
	}

	return gpuTime;
}
