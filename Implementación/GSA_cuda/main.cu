#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include "cudaBase.h"

// Como compilar:
// nvcc -std=c++11 -arch=sm_35 main.cu

using namespace std;

#include <cuda_runtime.h>
#include <cuda.h>
#include <curand_kernel.h>


static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
	if (err != cudaSuccess)
	{
		fprintf(stderr, "%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n", msg, file_name, line_number, cudaGetErrorString(err));
		std::cin.get();
		exit(EXIT_FAILURE);
	}
}

#define SAFE_CALL(call, msg) _safe_cuda_call((call),(msg),__FILE__, __LINE__)


class CudaTime {
	cudaEvent_t start, stop;

public:
	CudaTime() {
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}
	void record() {
		cudaEventRecord( start, 0 );
	}
	void stopAndPrint(const char* msg) {
		cudaEventRecord( stop, 0 );
		cudaEventSynchronize( stop );
		float elapsedTime;
		cudaEventElapsedTime( &elapsedTime, start, stop);
		printf( "Elapsed time on %s: %3.1f ms\n", msg , elapsedTime );
	}
};



__device__
float Ackley(float* X, int dim) //
{
	float a = 20.0f, b = 0.2f, c = 2 * M_PI, sum = 0.0f, cosenos = 0.0f;
	for (int i = 0; i < dim; ++i)
	{
		sum += X[i] * X[i];
		cosenos += cosf(c * X[i]);
	}
	return -a * expf(-b * sqrtf(sum / dim)) - expf(cosenos / dim) + a + expf(1);
}

__device__
float Schwefel(float* X, int dim)
{
	float sum = 0.0f;
	for (int i = 0; i < dim; ++i)
		sum +=  X[i] * sinf(sqrtf(fabsf(X[i])));

	return 418.9829 * dim - sum;
}

__device__
float Funcion_3(float* X, int dim)
{
	float sum = 0.0f;
	for (int i = 0; i < dim; ++i)
		sum += X[i] * X[i];

	return 0.5 - (   powf(sinf(sqrtf(sum)), 2) - 0.5   ) / powf(1.0 + 0.001 * sum, 2);
}




__global__ void GSA_iteration_fitness(float** position, float* fitness, int funcion, int numero_poblacion, int dim) {

	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	const int index = yIndex * blockDim.x * gridDim.x + xIndex;

	if (index < numero_poblacion)
	{
		switch (funcion) {
		case 1:
			fitness[index] = Ackley(position[index], dim);
			break;
		case 2:
			fitness[index] = Schwefel(position[index], dim);
			break;
		case 3:
			fitness[index] = Funcion_3(position[index], dim);
			break;
		}
	}
}


__device__
float* getF(float* i_position, float i_Mp, float* j_position, float j_Ma, int dimension)
{
	float R = 0.0f;
	// int dim = i.position.size();
	for (size_t component = 0; component < dimension; component++)
		R += pow(i_position[component] - j_position[component], 2);

	R = sqrtf(R);

	float e = 2.2204e-016;//algun valor pequeño
	float sub_F = i_Mp * j_Ma / (R + e);

	float* Force = new float[dimension * sizeof(float)];

	for (size_t component = 0; component < dimension; component++)
		Force[component] = sub_F * (j_position[component] - i_position[component]);

	return Force;
}

__global__ void initRandomSeed( curandState *devState, unsigned int seed ,  int numero_poblacion) {

	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	const int index = yIndex * blockDim.x * gridDim.x + xIndex;
	if (index < numero_poblacion)
		curand_init( seed, index, 0, &devState[index] );
}


__global__ void GSA_iteration_move(float** position, float** A, float** V, float** F,
                                   float* fitness, float* M, float* sum,
                                   float G, int numero_poblacion, int dimension, int iteraciones, int min, int max,
                                   float best, float worst, curandState *devState) {

	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	const int index = yIndex * blockDim.x * gridDim.x + xIndex;

	if (index < numero_poblacion)
	{

		float m = (fitness[index] - worst) / (best - worst); //Equation 15

		atomicAdd( sum, m );
		__syncthreads();

		M[index] = m / *sum;     //Equation 16

		//(e)Calculation of the total force in different directions.
		int l;
		if (iteraciones < numero_poblacion) l = iteraciones;
		else l = numero_poblacion;


		//vaciamos forces
		for (size_t k = 0; k < dimension; k++)
			F[index][k] = 0;

		//mejora, solo los mejoras van a influir.
		for (size_t j = 0; j < l; j++)
		{
			if (index != j)
			{
				float* Force = getF(position[index], M[index] , position[j], M[j], dimension);

				for (size_t k = 0; k < dimension; k++)
					F[index][k] += curand_uniform(&devState[index]) * G * Force[k];
				free(Force);
			}
		}


		//(f)Calculation of acceleration and velocity.
		for (size_t i = 0; i < dimension; i++)
		{
			if (M[index] == 0)
				A[index][i] = 0;
			else
				A[index][i] = F[index][i] / M[index];     //Equation 10
		}

		for (size_t i = 0; i < dimension; i++)
			V[index][i] = curand_uniform(&devState[index]) * V[index][i] + A[index][i];  //Equation 11


		//(g)Updating agents’ position.
		for (size_t i = 0; i < dimension; i++)
		{
			// position[index][i] = std::min<float>(position[index][i] + V[index][i], max);
			if (position[index][i] +  V[index][i] < max)
				position[index][i] = position[index][i] +  V[index][i];
			else
				position[index][i] = max;

			// position[index][i] = std::max<float>(position[index][i], min);  //Equation 12
			if (position[index][i] > min)
				position[index][i] = position[index][i];
			else
				position[index][i] = min;
		}
	}
}

void GSA_cu(int funcion, unsigned int numero_poblacion, unsigned int dimension, float minimo, float maximo, bool minimizar,
            int iteraciones, float G) {

	//Initialize
	std::random_device generador;
	std::uniform_real_distribution<float> distribucion(minimo, maximo);

	float** position = (float **) malloc( numero_poblacion * sizeof(float*)) ;
	for (int i = 0; i < numero_poblacion; ++i) {
		position[i] = (float *)malloc(dimension * sizeof(float));
		for (size_t j = 0; j < dimension; j++)
			position[i][j] = distribucion(generador);
	}

	float bestFitnessSoFar;
	std::vector<float> bestPositionSoFar(dimension);

	if (minimizar) bestFitnessSoFar = 1000000000000.0;
	else bestFitnessSoFar = -1000000000000.0;

	float bestFitness ;
	float worstFitness ;
	// int bestFitness_idx ;

	if (minimizar) {
		bestFitness = 1000000000;
		worstFitness = -1000000000;
	} else {
		bestFitness = -1000000000;
		worstFitness = 1000000000;
	}

	float G_step = G / iteraciones;



	//(b)Randomized initialization.


	/*** Copy Position HOST TO DEVICE ***/
	float** dev_position = 0;
	float*  dev_temp_position[numero_poblacion];

	// first create top level pointer
	SAFE_CALL(cudaMalloc(&dev_position,  sizeof(float*)  * numero_poblacion), "CUDA Malloc Failed");

	// then create child pointers on host, and copy to device, then copy image
	for (int i = 0; i < numero_poblacion; i++)
	{
		SAFE_CALL(cudaMalloc(&dev_temp_position[i], dimension * sizeof(float) ), "CUDA Memset Failed");

		SAFE_CALL(cudaMemcpy(&(dev_position[i]), &(dev_temp_position[i]), sizeof(float *), cudaMemcpyHostToDevice), "CUDA Memset Failed");//copy child pointer to device
		SAFE_CALL(cudaMemcpy(dev_temp_position[i], position[i], dimension * sizeof(float), cudaMemcpyHostToDevice), "CUDA Memset Failed"); // copy image to device
	}

	/*** end Copy Position ***/


	/*** Copy Velocity HOST TO DEVICE ***/
	float** dev_V = 0;
	float*  dev_temp_V[numero_poblacion];

	// first create top level pointer
	SAFE_CALL(cudaMalloc(&dev_V,  sizeof(float*)  * numero_poblacion), "CUDA Malloc Failed");

	// then create child pointers on host, and copy to device, then copy image
	for (int i = 0; i < numero_poblacion; i++)
	{
		SAFE_CALL(cudaMalloc(&dev_temp_V[i], dimension * sizeof(float) ), "CUDA Memset Failed");
		SAFE_CALL(cudaMemcpy(&(dev_V[i]), &(dev_temp_V[i]), sizeof(float *), cudaMemcpyHostToDevice), "CUDA Memset Failed");//copy child pointer to device
	}

	/*** end Copy Velocity ***/

	/*** Copy Acceleration HOST TO DEVICE ***/
	float** dev_A = 0;
	float*  dev_temp_A[numero_poblacion];

	// first create top level pointer
	SAFE_CALL(cudaMalloc(&dev_A,  sizeof(float*)  * numero_poblacion), "CUDA Malloc Failed");

	// then create child pointers on host, and copy to device, then copy image
	for (int i = 0; i < numero_poblacion; i++)
	{
		SAFE_CALL(cudaMalloc(&dev_temp_A[i], dimension * sizeof(float) ), "CUDA Memset Failed");
		SAFE_CALL(cudaMemcpy(&(dev_A[i]), &(dev_temp_A[i]), sizeof(float *), cudaMemcpyHostToDevice), "CUDA Memset Failed");//copy child pointer to device
	}

	/*** end Copy Acceleration ***/

	/*** Copy force HOST TO DEVICE ***/
	float** dev_F = 0;
	float*  dev_temp_F[numero_poblacion];

	// first create top level pointer
	SAFE_CALL(cudaMalloc(&dev_F,  sizeof(float*)  * numero_poblacion), "CUDA Malloc Failed");

	// then create child pointers on host, and copy to device, then copy image
	for (int i = 0; i < numero_poblacion; i++)
	{
		SAFE_CALL(cudaMalloc(&dev_temp_F[i], dimension * sizeof(float) ), "CUDA Memset Failed");
		SAFE_CALL(cudaMemcpy(&(dev_F[i]), &(dev_temp_F[i]), sizeof(float *), cudaMemcpyHostToDevice), "CUDA Memset Failed");//copy child pointer to device
	}

	/*** end Copy force ***/


	// Device variables
	curandState *devState;
	cudaMalloc( (void **)&devState, numero_poblacion * sizeof(curandState) );

	float* dev_fitness;
	float* dev_M;
	float* fitness = new float[numero_poblacion * sizeof(float)];

	SAFE_CALL(cudaMalloc(&dev_fitness, numero_poblacion * sizeof(float) ), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc(&dev_M, numero_poblacion * sizeof(float) ), "CUDA Malloc Failed");

	//Allocate mempry for the sum
	float* dev_sum;
	// float* sum = (float*)malloc(sizeof(float));
	cudaMalloc((void**)&dev_sum, sizeof(float));
	cudaMemset(dev_sum, 0, sizeof(float));



	//Specify a reasonable block size
	const dim3 block(4, 4);

	//Calculate grid size to cover the whole image
	const dim3 grid( ( ceil(sqrt(numero_poblacion)) + block.x - 1) / block.x,
	                 ( ceil(sqrt(numero_poblacion)) + block.y - 1) / block.y);  //implicit cast to int ceil

	initRandomSeed <<< grid, block>>>(devState, (unsigned int)time(NULL), numero_poblacion);

	for (int _ = 0; _ < iteraciones - 1; ++_)
	{

		GSA_iteration_fitness <<< grid, block>>>(dev_position, dev_fitness, funcion, numero_poblacion, dimension);
		cudaDeviceSynchronize();

		SAFE_CALL(cudaMemcpy(fitness, dev_fitness, numero_poblacion * sizeof(float) , cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

		for (int i = 0; i < numero_poblacion; ++i)
		{
			if (minimizar) {
				if (fitness[i] < bestFitness){
					bestFitness = fitness[i];
					// bestFitness_idx = i;
				}

				if (fitness[i] > worstFitness)
					worstFitness = fitness[i];
			} else {
				if (fitness[i] > bestFitness){
					bestFitness = fitness[i];
					// bestFitness_idx = i;
				}

				if (fitness[i] < worstFitness)
					worstFitness = fitness[i];
			}
		}
		if (minimizar)
		{
			if (bestFitness < bestFitnessSoFar) 
				bestFitnessSoFar = bestFitness;
		}	else {
			if (bestFitness > bestFitnessSoFar) 
				bestFitnessSoFar = bestFitness;
		}

		G -= G_step;

		GSA_iteration_move <<< grid, block>>>(dev_position, dev_A, dev_V, dev_F, dev_fitness, dev_M, dev_sum,  G, numero_poblacion,
		                                      dimension,  iteraciones, minimo, maximo, bestFitness, worstFitness, devState);
		cudaDeviceSynchronize();
		cudaMemset(dev_sum, 0, sizeof(float));
	}
	cout<<endl<<"Result: "<< bestFitnessSoFar<<endl;
}



int main()
{
	int numero_poblacion = 60, dimension = 2;
	float G = 18;
	bool minimizar = true;
	int iteraciones = 50;
	CudaTime tiempo;
	tiempo.record();

	for (int i = 0; i < 1; ++i)
	{
		float minimo = -32.768f, maximo = 32.768f;
		GSA_cu(1, numero_poblacion, dimension, minimo, maximo, minimizar, iteraciones, G);
	}

	// cout << "All good" << endl;


	// for (int i = 0; i < 1; ++i)
	// {
	//  float minimo = -500.0f, maximo = 500.0f;
	//  GSA_cu(2, numero_poblacion, dimension, minimo, maximo, minimizar, iteraciones, G);
	// }


	// for (int i = 0; i < 1; ++i)
	// {
	//  minimizar = false;
	//  float minimo = -100.0f, maximo = 100.0f;
	//  GSA_cu(3, numero_poblacion, dimension, minimo, maximo, minimizar, iteraciones, G);
	// }


	tiempo.stopAndPrint("GSA");

	return 0;
}