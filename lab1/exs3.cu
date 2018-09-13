#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h> 

#include <iostream> 
#include <cstdlib> 
#include <ctime> 

struct sum_functor
{
	__host__  __device__
		float operator () (const  float & x, const  float & y) const {
		return  x + y;
	}
};

thrust::device_vector<int> multi_fast(thrust::device_vector < int > & X, thrust::device_vector < int > & Y)
{
	// Y <-  X + Y
	thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), sum_functor());
	return Y;
}

int main(int argc, char* argv[])
{

	for (int i = 10; i < 100000000; i *= 10) {
		thrust::device_vector<int> a(i);
		thrust::device_vector<int> b(i);

		thrust::fill(a.begin(), a.end(), 3);
		thrust::fill(b.begin(), b.end(), 1);


		cudaEvent_t start, stop;
		float gpuTime = 0.0f;

		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start, 0);

		multi_fast(a, b);

		cudaEventRecord(stop, 0);

		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&gpuTime, start, stop);
		std::cout<<gpuTime<<std::endl;
		
		


		//for (int i = 0; i < b.size(); i++)
		//	std::cout << "b[" << i << "] = " << b[i] << std::endl;
	}
	return 0;
}