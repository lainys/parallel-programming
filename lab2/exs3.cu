#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <fstream>


float cudaParallel(int **a, int **b, int **c, int n);
float cudaParallelShare(int **a, int **b, int **c, int n);
float check(int** a, int**b, int**c, int n, int k);
float checkCpu(int** a, int**b, int**c, int n, int k);
float checkShare(int** a, int**b, int**c, int n, int k);


//cpu
int iter(int* a, int* b, int n) {
	int sum = 0;
	for (int i = 0; i < n; i++) {
		sum += a[i] * b[i];
	}
	return sum;
}

int** t(int** a, int n) {
	int** res = new int*[n];

	for (int i = 0; i < n; i++) {
		res[i] = new int[n];

		for (int j = 0; j < n; j++) {
			res[i][j] = a[j][i];
		}
	}
	return res;
}

float cpuMulti(int** a, int** b, int** c, int n) {
	cudaEvent_t start, stop;
	float gpuTime = 0.0f;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
	//----


	///
	b = t(b, n);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			c[i][j] = iter(a[i], b[j], n);
		}
	}
	///

	cudaDeviceSynchronize();
	//----
	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuTime, start, stop);

	//printf("N = %d, time spent executing by the GPU: %.5f ms\n",N, gpuTime);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return gpuTime;
}

__global__ void gpuMulti(int *a, int *b, int *c, int n)
{
	int x = threadIdx.y;
	int y = threadIdx.x;

	int sum = 0;
	for (int i = 0; i < n; i++) {
		sum += a[x*n + i] * b[y*n + i];
	}

	c[x*n + y] = sum;

}
#define BLOCK_SIZE 16
__global__ void kernel(int * a, int * b, int n, int * c)
{
	int bx = blockIdx.x;  // индексы блока 
	int by = blockIdx.y;  // 

	int tx = threadIdx.x;  // индексы нити внутри блока 
	int ty = threadIdx.y;  // 

	int	aBegin = n * BLOCK_SIZE * by;
	int	aEnd = aBegin + n - 1;
	int aStep = BLOCK_SIZE;
	int	bBegin = bx * BLOCK_SIZE;
	int bStep = BLOCK_SIZE * n;
	float	sum = 0.0f;
	for (int ia = aBegin, ib = bBegin; ia <= aEnd; ia += aStep, ib += bStep)
	{
		__shared__ float	as[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float	bs[BLOCK_SIZE][BLOCK_SIZE];
		as[ty][tx] = a[ia + n * ty + tx];
		bs[ty][tx] = b[ib + n * ty + tx];
		__syncthreads(); // Убедимся, что подматрицы полностью загружены 
		for (int k = 0; k < BLOCK_SIZE; k++)
			sum += as[ty][k] * bs[k][tx];
		__syncthreads(); // Убедимся, что подматрицы никому больше не нужны 
	}
	c[n * BLOCK_SIZE * by + BLOCK_SIZE * bx + n * ty + tx] = sum;
}


int main()
{
	int k = 10;

	float *ans1 = new float[k];
	float *ans2 = new float[k];
	float *ans3 = new float[k];
	int *var_N = new int[k];

	for (int i = 0, N = 20; i < k; i++, N += 20) {

		var_N[i] = N;

		int**a = new int*[N];
		int**b = new int*[N];
		int**c = new int*[N];
		
		for (int i = 0; i < N; i++) {
			a[i] = new int[N];
			b[i] = new int[N];
			c[i] = new int[N];
			for (int j = 0; j < N; j++) {
				if (i == j) {
					a[i][j] = 3;
					b[i][j] = 3;
					c[i][j] = 3;
				}
				else {
					a[i][j] = 0;
					b[i][j] = 0;
					c[i][j] = 0;
				}
			}
		}
		

		std::cout << i << " ";
		ans1[i] = checkCpu(a, b, c, N, 10);
		std::cout << i << " ";
		ans2[i] = check(a, b, c, N, 10);
		std::cout << i << " ";
		ans3[i] = checkShare(a, b, c, N, 10);
		std::cout << i << " ";
		/*
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				std::cout << c[i][j] << " ";
			}
			std::cout << std::endl;
		}*/

		std::cout << std::endl;
		delete a, b, c;
	}


	std::ofstream out("text3.txt", 'w');

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

	for (int i = 0; i < k; i++) {
		out << ans3[i] << " ";
	}
	out << std::endl;

	out.close();

	delete ans1, ans2, ans3, var_N;

	return 0;
}

float check(int** a, int** b, int**c, int n, int k) {
	float res = 0;

	for (int i = 0; i < k; i++) {
		res += cudaParallel(a, b, c, n);
	}

	return res / (k * 1.0f);
}

float checkCpu(int** a, int** b, int**c, int n, int k) {
	float res = 0;

	for (int i = 0; i < k; i++) {
		res += cpuMulti(a, b, c, n);
	}

	return res / (k * 1.0f)*1000;
}



float checkShare(int** a, int** b, int**c, int n, int k) {
	float res = 0;

	for (int i = 0; i < k; i++) {
		res += cudaParallelShare(a, b, c, n);
	}

	return res / (k * 1.0f);
}



float cudaParallel(int **a, int **b, int **c, int n) {

	int** t_b = t(b, n);

	int* copy_a = new int[n*n];
	int* copy_b = new int[n*n];
	int* copy_c = new int[n*n];


	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			copy_a[i*n + j] = a[i][j];
			copy_b[i*n + j] = t_b[i][j];
			copy_c[i*n + j] = c[i][j];
		}
	}



	int* dev_a;
	int* dev_b;
	int* dev_c;

	// init memory
	cudaMalloc((void**)&dev_a, n *n * sizeof(int));
	cudaMalloc((void**)&dev_b, n *n * sizeof(int));
	cudaMalloc((void**)&dev_c, n *n * sizeof(int));

	// copy to device
	cudaMemcpy(dev_a, copy_a, n*n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, copy_b, n*n * sizeof(int), cudaMemcpyHostToDevice);


	//----
	cudaEvent_t start, stop;
	float gpuTime = 0.0f;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
	//----


	gpuMulti << <1, dim3(n, n) >> > (dev_a, dev_b, dev_c, n);


	cudaDeviceSynchronize();
	//----
	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuTime, start, stop);

	//printf("N = %d, time spent executing by the GPU: %.5f ms\n",N, gpuTime);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	//----

	cudaMemcpy(copy_c, dev_c, n*n * sizeof(int), cudaMemcpyDeviceToHost);


	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);


	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			c[i][j] = copy_c[i*n + j];
		}
	}

	return gpuTime;
}


float cudaParallelShare(int **a, int **b, int **c, int n) {

	int** t_b = t(b, n);

	int* copy_a = new int[n*n];
	int* copy_b = new int[n*n];
	int* copy_c = new int[n*n];


	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			copy_a[i*n + j] = a[i][j];
			copy_b[i*n + j] = t_b[i][j];
			copy_c[i*n + j] = c[i][j];
		}
	}



	int* dev_a;
	int* dev_b;
	int* dev_c;

	// init memory
	cudaMalloc((void**)&dev_a, n *n * sizeof(int));
	cudaMalloc((void**)&dev_b, n *n * sizeof(int));
	cudaMalloc((void**)&dev_c, n *n * sizeof(int));

	// copy to device
	cudaMemcpy(dev_a, copy_a, n*n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, copy_b, n*n * sizeof(int), cudaMemcpyHostToDevice);


	//----
	cudaEvent_t start, stop;
	float gpuTime = 0.0f;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
	//----


	kernel << < dim3(n/BLOCK_SIZE, n/BLOCK_SIZE), dim3(BLOCK_SIZE, BLOCK_SIZE) >> > (dev_a, dev_b, n, dev_c);// , n);


	cudaDeviceSynchronize();
	//----
	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuTime, start, stop);

	//printf("N = %d, time spent executing by the GPU: %.5f ms\n",N, gpuTime);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	//----

	cudaMemcpy(copy_c, dev_c, n*n * sizeof(int), cudaMemcpyDeviceToHost);


	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);


	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			c[i][j] = copy_c[i*n + j];
		}
	}

	return gpuTime;
}
