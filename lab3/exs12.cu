#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <fstream>
#include <ctime>

#define LOG2_WARP_SIZE  5 
#define WARP_SIZE       32 
#define WARP_N          8  
#define NUM_BINS		256

float checkGPU(int* a, unsigned int* hist, int N, int k);
float checkCPU(int* a, unsigned int* h, int N, int k);
unsigned int minCPU(int* a, unsigned int* h, int N, int bins);

__device__ inline void addByte(volatile unsigned * warp_hist, unsigned data, unsigned ttag)
{
	unsigned count;
	do {
		count = warp_hist[data] & 0x07FFFFFFU;

		count = ttag | (count + 1);
		warp_hist[data] = count;
	} while (warp_hist[data] != count);
}

__global__ void histogramKernel(unsigned * result, int * data, int n) {
	__shared__ unsigned hist[NUM_BINS*WARP_N]; //1536 элементов
											   // очистить счетчики гистограмм
	for (int i = 0; i < NUM_BINS / WARP_SIZE; i++)
		hist[threadIdx.x + i * WARP_N*WARP_SIZE/*число нитей в блоке=192*/] = 0;

	int warp_base = (threadIdx.x >> LOG2_WARP_SIZE) * NUM_BINS;
	unsigned ttag = threadIdx.x << (32 - LOG2_WARP_SIZE); // получить id для данной нити

	__syncthreads();
	int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
	int numThreads = blockDim.x * gridDim.x;
	for (int i = global_tid; i < n; i += numThreads)
	{
		unsigned data4 = data[i];
		addByte(hist + warp_base, (data4 >> 0) & 0xFFU, ttag);
		addByte(hist + warp_base, (data4 >> 8) & 0xFFU, ttag);
		addByte(hist + warp_base, (data4 >> 16) & 0xFFU, ttag);
		addByte(hist + warp_base, (data4 >> 24) & 0xFFU, ttag);
	}
	__syncthreads();
	// объединить гистограммы данного блока и записать результат в глобальную память
	// 192 нити суммируют данные до 256 элементов гистограмм
	for (int bin = threadIdx.x; bin < NUM_BINS; bin += (WARP_N*WARP_SIZE))
	{
		unsigned sum = 0;
		for (int i = 0; i < WARP_N; i++)
			sum += hist[bin + i * NUM_BINS] & 0x07FFFFFFU;
		result[blockIdx.x * NUM_BINS + bin] = sum;
	}
}
// объединить гистограммы, один блок на каждый NUM_BINS элементов
__global__ void mergeHistogramKernel(unsigned * out_histogram, unsigned * partial_histograms, int histogram_count)
{
	unsigned  sum = 0;
	for (int i = threadIdx.x; i < histogram_count; i += NUM_BINS)
		sum += partial_histograms[blockIdx.x + i * NUM_BINS];

	__shared__ unsigned data[NUM_BINS];
	data[threadIdx.x] = sum;

	for (unsigned stride = NUM_BINS / 2; stride > 0; stride >>= 1)
	{
		__syncthreads();
		if (threadIdx.x < stride) data[threadIdx.x] += data[threadIdx.x + stride];
	}
	if (threadIdx.x == 0) out_histogram[blockIdx.x] = data[0];
}
float histogram(unsigned int * histogram, void * data, unsigned byte_сount)
{
	int num_partials = byte_сount / (WARP_N*WARP_SIZE) + 1;
	unsigned *partial_histograms = nullptr;
	unsigned int h[NUM_BINS] = { 0 };
	int *pdata = (int*)data;

	unsigned int* dev_h = nullptr;
	//выделить память под гистограммы блока
	cudaMalloc((void**)&partial_histograms, num_partials*NUM_BINS * sizeof(unsigned));
	cudaMalloc((void**)&dev_h, NUM_BINS * sizeof(unsigned int));


	int* dev_data;
	cudaMalloc((void**)&dev_data, byte_сount * sizeof(int));
	cudaMemcpy(dev_data, data, byte_сount * sizeof(int), cudaMemcpyHostToDevice);


	cudaEvent_t start, stop;
	float gpuTime = 0.0f;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	
	// построить гистограмму для каждого блока
	histogramKernel << <dim3(num_partials), dim3(WARP_N*WARP_SIZE) >> > (partial_histograms, dev_data, byte_сount);

	//объдинить гистограммы отдельных блоков вместе
	mergeHistogramKernel << <dim3(NUM_BINS), dim3(NUM_BINS) >> > (dev_h, partial_histograms, num_partials);
	// освободить выделенную память
	//----
	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuTime, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaMemcpy(histogram, dev_h, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

	cudaFree(partial_histograms);
	cudaFree(dev_h);
	return gpuTime;
}


void init(int* a, int N) {
	for (int i = 0; i < N; i++) {
		a[i] = 250;
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
		unsigned int* h = new unsigned int[NUM_BINS];
		init(a, N);

		ans1[i] = checkCPU(a, h, N, 100);
		ans2[i] = checkGPU(a, h, N, 100);


		std::cout << i <<" "<<ans2[i]<< std::endl;

		delete a;
	}

	std::ofstream out("text2.txt", 'w');

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

float checkGPU(int* a, unsigned int* hist, int N, int k) {
	float res = 0;

	for (int i = 0; i < k; i++) {
		res += histogram(hist, a, N);
	}

	return res / (k * 1.0f);
}

float checkCPU(int* a, unsigned int* h, int N, int k) {
	unsigned int res = 0;

	for (int i = 0; i < k; i++) {
		res += minCPU(a, h, N, NUM_BINS);
	}

	return res / (k * 1000.0f);
}


unsigned int minCPU(int* a, unsigned int* h, int N, int bins) {

	for (int i = 0; i < bins; i++) {
		h[i] = 0;
	}
	//----
	/*cudaEvent_t start, stop;
	float gpuTime = 0.0f;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);*/

	unsigned int start_time = clock(); // начальное время
									   // здесь должен быть фрагмент кода, время выполнения которого нужно измерить

	//----

	for (int i = 1; i < N; i++) {
		if (a[i] < bins) {
			h[a[i]]++;
		}
	}
	unsigned int end_time = clock(); // конечное время
	unsigned int search_time = end_time - start_time;
	//----
	/*cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuTime, start, stop);

	//printf("N = %d, time spent executing by the GPU: %.5f ms\n",N, gpuTime);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);*/
	//----

	return search_time;
}