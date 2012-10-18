#include "mlp/cuda/DeviceLayer.h"

#define CUDA_RAND_MAX 4294967295

namespace ParallelMLP
{

__device__
float d_random(curandState* state);

__device__
float d_activate(float x);

__device__
float d_derivate(float y);

//===========================================================================//

DeviceLayer::DeviceLayer(uint inUnits, uint outUnits)
{
	init(inUnits, outUnits);
}

//===========================================================================//

__global__
void initRandState(curandState* state, int seed, uint connUnits)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < connUnits)
		curand_init(seed + i, 0, 0, &state[i]);
}

//===========================================================================//

void DeviceLayer::init(uint inUnits, uint outUnits)
{
	this->inUnits = inUnits + 1;
	this->outUnits = outUnits;
	this->connUnits = (inUnits + 1) * outUnits;
	this->connBlocks = connUnits / TPB + 1;
	this->outBlocks = outUnits / TPB + 1;

	cudaMalloc(&weights, connUnits * sizeof(float));
	cudaMalloc(&gradient, outUnits * sizeof(float));
	cudaMalloc(&funcSignal, (outUnits + 1) * sizeof(float));
	cudaMalloc(&errorSignal, inUnits * sizeof(float));
	cudaMalloc(&state, connUnits * sizeof(curandState));

	float aux = 1;

	cudaMemcpy(&funcSignal[outUnits], &aux, sizeof(float),
			cudaMemcpyHostToDevice);

	initRandState<<<connBlocks, TPB>>>(state, rand(), connUnits);
}

//===========================================================================//

DeviceLayer::~DeviceLayer()
{

}

//===========================================================================//

__global__
void randomizeWeight(float* weights, curandState* state, uint connUnits)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < connUnits)
		weights[i] = d_random(&state[i]);
}

//===========================================================================//

void DeviceLayer::randomize()
{
	randomizeWeight<<<connBlocks, TPB>>>(weights, state, connUnits);
}

//===========================================================================//

__global__
void feedforwardSum(const float* input, float* weights, uint inUnits,
		uint connUnits, float* funcSignal)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = i % inUnits;
	int k = i / inUnits;

	if (i < connUnits)
		funcSignal[k] += weights[i] * input[j];
		//atomicAdd(&funcSignal[k], weights[i] * input[j]);
}

//===========================================================================//

__global__
void feedforwardActivate(float* weights, uint outUnits, float* funcSignal)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < outUnits)
		funcSignal[i] = d_activate(funcSignal[i]);
}

//===========================================================================//

void DeviceLayer::feedforward(const float* input)
{
	this->input = input;

	// Inicializa o sinal funcional
	cudaMemset(funcSignal, 0, outUnits * sizeof(float));

	// Calcula as somas ponderadas das entradas
	feedforwardSum<<<connBlocks, TPB>>>(input, weights, inUnits, connUnits,
			funcSignal);

	// Ativa as saídas de cada neurônio
	feedforwardActivate<<<outBlocks, TPB>>>(weights, outUnits, funcSignal);
}

//===========================================================================//

__global__
void feedbackDerivate(const float* signal, float* funcSignal, uint outUnits,
		float* gradient)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < outUnits)
		gradient[i] = d_derivate(funcSignal[i]) * signal[i];
}

//===========================================================================//

__global__
void feedbackSum(const float* input, float* gradient, float learning,
		uint inUnits, uint connUnits, float* weights, float* errorSignal)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = i % inUnits;
	int k = i / inUnits;

	if (i < connUnits)
	{
		weights[i] += learning * gradient[k] * input[j];
		errorSignal[j] += gradient[k] * weights[i];
		//atomicAdd(&errorSignal[j], gradient[k] * weights[i]);
	}
}

//===========================================================================//

void DeviceLayer::feedback(const float* signal, float learning)
{
	// Inicializa o sinal funcional
	cudaMemset(errorSignal, 0, (inUnits - 1) * sizeof(float));

	// Calcula o gradiente
	feedbackDerivate<<<outBlocks, TPB>>>(signal, funcSignal, outUnits,
			gradient);

	// Realiza a atualização dos pesos e cálculo do sinal de erro
	feedbackSum<<<connBlocks, TPB>>>(input, gradient, learning, inUnits,
			connUnits, weights, errorSignal);
}

//===========================================================================//

uint DeviceLayer::getInUnits()
{
	return inUnits;
}

//===========================================================================//

uint DeviceLayer::getOutUnits()
{
	return outUnits;
}

//===========================================================================//

float* DeviceLayer::getFuncSignal()
{
	return funcSignal;
}

//===========================================================================//

float* DeviceLayer::getErrorSignal()
{
	return errorSignal;
}

//===========================================================================//

__device__
float d_random(curandState* state)
{
	float r = curand(state) / (float) CUDA_RAND_MAX;
	return 2 * r - 1;
}

//===========================================================================//

__device__
float d_activate(float x)
{
	return tanh(x);
}

//===========================================================================//

__device__
float d_derivate(float y)
{
	return (1 - y) * (1 + y);
}

//===========================================================================//

}

