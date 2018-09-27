# include <time.h>
# include <stdlib.h>
# include <stdio.h>
# include <string.h>
# include <cuda.h>
# include <ctime>
#include <cuda_runtime.h>
#include "./common/inc/helper_image.h"


unsigned int width = 512, height = 512;

texture<unsigned char, 2, cudaReadModeElementType> g_Texture;
__global__ void Bilinear(unsigned char * dest, float factor, unsigned  int w, unsigned  int h)
{
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	int tidy = threadIdx.y + blockIdx.y * blockDim.y;
	// проверка, что текущие индексы не выходят за границы изображения
	if (tidx < w && tidy < h) {
		float center = tidx / factor;
		unsigned  int start = (unsigned  int)center;
		unsigned  int stop = start + 1.0f;
		float t = center - start;
		unsigned char a = tex2D(g_Texture, tidy + 0.5f, start + 0.5f);
		unsigned char b = tex2D(g_Texture, tidy + 0.5f, stop + 0.5f);
		float linear = a + (b - a)*t;//lerp(a, b, t);
		dest[tidx + tidy * w] = (int)(linear);
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

	int factor = 2;

	char * src_path = "lena.pgm";
	char * d_result_path = "lena_bisfil.pgm";

	loadImage(src_path, &h_pixels, &width, &height);

	printf("Image size %dx%d\n", width, height);

	int image_size = sizeof(unsigned char) * width * height;

	h_result_pixels = (unsigned char *)malloc(image_size);
	cudaMalloc((void **)& d_result_pixels, image_size);


	cudaArray * array;
	

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar1>();
	cudaMallocArray(&array, &desc, width, height); 
	cudaMemcpyToArray(array, 0, 0, h_pixels, image_size,cudaMemcpyHostToDevice);



	cudaError_t error = cudaBindTextureToArray(&g_Texture, array, &desc);

	if (cudaSuccess != error) {
		printf("ERROR: Failed to bind texture.\n");
		exit(-1);
	}
	else {
		printf("Texture was successfully binded\n");
	}


	int n = 16;
	dim3 block(n, n);
	dim3 grid(width / n, height / n);



	Bilinear << < grid, block >> >( d_result_pixels,factor, width, height);

	cudaMemcpy(h_result_pixels, d_result_pixels, image_size, cudaMemcpyDeviceToHost);
	saveImage(d_result_path, h_result_pixels, width, height);
	cudaUnbindTexture(&g_Texture);

	cudaFree(d_result_pixels);

	//delete rs, ans1;

	return 0;
}
