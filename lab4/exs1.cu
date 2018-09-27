# include <time.h>
# include <stdlib.h>
# include <stdio.h>
# include <string.h>
# include <cuda.h>
# include <ctime>
#include <cuda_runtime.h>
#include "./common/inc/helper_image.h"

float checkGPU(unsigned char * d_result_pixels, int radius, int k);
float cudaPallel(unsigned char * d_result_pixels, int radius);

texture<unsigned char, 2, cudaReadModeElementType> g_Texture;
unsigned int width = 512, height = 512;


__global__  void BoxBlur_kernel(unsigned char * pDst, int radius, int w, int h)
{
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	int tidy = threadIdx.y + blockIdx.y * blockDim.y;
	// проверка, что текущие индексы не выходят за границы изображения 
	if (tidx < w && tidy < h)
	{
		unsigned int r = 0;
		for (int ir = -radius; ir <= radius; ir++)
			for (int ic = -radius; ic <= radius; ic++)
			{
				r += tex2D(g_Texture, tidx + 0.5f + ic, tidy + 0.5f + ir);
			}
		//нормализация полученных результатов 
		r /= ((2 * radius + 1)*(2 * radius + 1));
		pDst[tidx + tidy * w] = (unsigned char)r;
	}
}

void loadImage(char *file, unsigned char** pixels, unsigned int * width, unsigned int * height)
{
	size_t file_length = strlen(file);

	if (!strcmp(&file[file_length - 3], "pgm"))
	{
		if (sdkLoadPGM<unsigned char>(file, pixels, width, height) != true)
		{
			printf("Failed to load PGM image file: %s\n", file);
			exit(EXIT_FAILURE);
		}
	}
	return;
}

void saveImage(char *file, unsigned char* pixels, unsigned int width, unsigned int  height)
{
	size_t file_length = strlen(file);
	if (!strcmp(&file[file_length - 3], "pgm"))
	{
		sdkSavePGM(file, pixels, width, height);
	}
	return;
}

int main(int argc, char ** argv)
{
	unsigned char * d_result_pixels;
	unsigned char * h_result_pixels;
	unsigned char * h_pixels = NULL;
	unsigned char * d_pixels = NULL;

	int radius = 8;

	char * src_path = "lena.pgm";
	char * d_result_path = "lena_box_blur.pgm";

	loadImage(src_path, &h_pixels, &width, &height);

	printf("Image size %dx%d\n", width, height);

	int image_size = sizeof(unsigned char) * width * height;

	h_result_pixels = (unsigned char *)malloc(image_size);
	cudaMalloc((void **)& d_pixels, image_size);
	cudaMalloc((void **)& d_result_pixels, image_size);
	cudaMemcpy(d_pixels, h_pixels, image_size, cudaMemcpyHostToDevice);


	cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar1>();
	cudaError_t error = cudaBindTexture2D(0, &g_Texture, d_pixels, &desc, width, height, width * sizeof(unsigned char));

	if (cudaSuccess != error) {
		printf("ERROR: Failed to bind texture.\n");
		exit(-1);
	}
	else {
		printf("Texture was successfully binded\n");
	}

	int N = radius - 1;

	int* rs = new int[N];
	float* ans1 = new float[N];

	for (int i = 1; i < radius; i++) {
		rs[i-1] = i;
		ans1[i-1] = checkGPU(d_result_pixels,i,100);
		std::cout << i << std::endl;
	}

	std::ofstream out("text1.txt", 'w');

	for (int i = 0; i < N; i++) {
		out << rs[i] << " ";
	}
	out << std::endl;


	for (int i = 0; i < N; i++) {
		out << ans1[i] << " ";
	}
	out << std::endl;


	out.close();
	

	cudaMemcpy(h_result_pixels, d_result_pixels, image_size, cudaMemcpyDeviceToHost);
	saveImage(d_result_path, h_result_pixels, width, height);
	cudaUnbindTexture(&g_Texture);

	cudaFree(d_pixels);
	cudaFree(d_result_pixels);

	delete rs, ans1;

	return 0;
}

float checkGPU(unsigned char * d_result_pixels,int radius, int k) {
	float time = 0;

	for (int i = 0; i < k; i++) {
		time += cudaPallel(d_result_pixels,radius);
	}

	return time / (1000.0f * k);
}

float cudaPallel(unsigned char * d_result_pixels,int radius) {

	int n = 16;
	dim3 block(n, n);
	dim3 grid(width / n, height / n);
	
	//----
	cudaEvent_t start, stop;
	float gpuTime = 0.0f;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
	//----
	BoxBlur_kernel << < grid, block >> >(d_result_pixels, radius, width, height);
	//negative_kernel << < grid, block >> >(d_result_pixels, width, height);

	//----
	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuTime, start, stop);

	//printf("N, time spent executing by the GPU: %.5f ms\n", gpuTime);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	//----
	/* CUDA method */

	return gpuTime;
}
