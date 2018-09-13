#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#define	N	(1024*1024)

__global__ void kernel(float * data)
{
	int   idx = blockIdx.x * blockDim.x + threadIdx.x;
	float x = 2.0f * 3.1415926f * (float)idx / (float)N;
	data[idx] = sinf(sqrtf(x));

}

int main(int argc, char *argv[])
{
	int		deviceCount;
	cudaDeviceProp	devProp;
	int blocks[3];
	int grids[3];

	cudaGetDeviceCount(&deviceCount);

	for (int device = 0; device < deviceCount; device++)
	{
		cudaGetDeviceProperties(&devProp, device);
		blocks[0] = devProp.maxThreadsDim[0];
		blocks[1] = devProp.maxThreadsDim[1];
		blocks[2] = devProp.maxThreadsDim[2];

		grids[0] = devProp.maxGridSize[0];
		grids[1] = devProp.maxGridSize[1];
		grids[2] = devProp.maxGridSize[2];
	}


	float *  a = (float*)malloc(N * sizeof(float));
	float * dev = nullptr;
	// выделить память на GPU
	cudaMalloc((void**)&dev, N * sizeof(float));

	cudaEvent_t start, stop;		//описываем переменные типа  cudaEvent_t 
	float       gpuTime = 0.0f;
	// создаем события начала и окончания выполнения ядра 
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	//привязываем событие start  к данному месту 
	cudaEventRecord(start, 0);
	// вызвать ядро 
	kernel <<<dim3(grids[0],grids[1],grids[2]), dim3(blocks[0],blocks[1],blocks[2]) >> > (dev);
	//привязываем событие stop  к данному месту 
	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);
	// запрашиваем время между событиями 
	cudaEventElapsedTime(&gpuTime, start, stop);
	printf("time spent executing by the GPU: %.5f ms\n", gpuTime);
	// уничтожаем созданные события 
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaMemcpy(a, dev, N * sizeof(float), cudaMemcpyDeviceToHost);
	// освободить выделенную память
	cudaFree(dev);

	free(a);


	return 0;
}
