#include "mlp/cuda/DeviceLayer.h"

namespace ParallelMLP
{

//===========================================================================//

DeviceLayer::DeviceLayer(uint inUnits, uint outUnits)
	: Layer(inUnits, outUnits)
{
	this->connBlocks = connUnits / TPB + 1;
	this->outBlocks = outUnits / TPB + 1;

	cudaMalloc(&weights, connUnits * sizeof(float));
	cudaMalloc(&bias, outUnits * sizeof(float));
	cudaMalloc(&gradient, outUnits * sizeof(float));
	cudaMalloc(&funcSignal, outUnits * sizeof(float));
	cudaMalloc(&errorSignal, inUnits * sizeof(float));

	// Cria um gerador de números aleatórios e seta a semente
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, rand());
}

//===========================================================================//

DeviceLayer::~DeviceLayer()
{
	cudaFree(weights);
	cudaFree(bias);
	cudaFree(gradient);
	cudaFree(funcSignal);
	cudaFree(errorSignal);
}

//===========================================================================//

void DeviceLayer::randomize()
{
	float* negone;
	float* randv;
	const float alpha = 1;

	// Cria um vetor com -1
	negone = new float[connUnits];
	for (uint i = 0; i < connUnits; i++)
		negone[i] = -1;

	// Aloca espaço para vetor de números aleatórios
	cudaMalloc(&randv, connUnits * sizeof(float));

	// Copia os -1 para o vetor de pesos e para o vetor de bias
	cudaMemcpy(weights, negone, connUnits * sizeof(float),
			cudaMemcpyHostToDevice);
	cudaMemcpy(bias, negone, outUnits * sizeof(float),
			cudaMemcpyHostToDevice);

	// Gera números de 0 a 1, os transforma para -1 e 1
	curandGenerateUniform(gen, randv, connUnits);
	cublasSaxpy(DeviceUtil::cublas, connUnits, &alpha, randv, 1, weights, 1);

	curandGenerateUniform(gen, randv, outUnits);
	cublasSaxpy(DeviceUtil::cublas, outUnits, &alpha, randv, 1, bias, 1);

	cudaFree(randv);
	delete[] negone;
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
}

//===========================================================================//

__global__
void feedforwardActivate(float* weights, float* bias, uint outUnits,
		float* funcSignal)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < outUnits)
		funcSignal[i] = ACTIVATE(bias[i] + funcSignal[i]);
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
	feedforwardActivate<<<outBlocks, TPB>>>(weights, bias, outUnits,
			funcSignal);
}

//===========================================================================//

__global__
void feedbackwardDerivate(const float* signal, float* funcSignal,
		uint outUnits, float* gradient, float* bias)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < outUnits)
	{
		gradient[i] = DERIVATE(funcSignal[i]) * signal[i];
		bias[i] += gradient[i];
	}
}

//===========================================================================//

__global__
void feedbackwardSum(const float* input, float* gradient, float learning,
		uint inUnits, uint connUnits, float* weights, float* errorSignal)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = i % inUnits;
	int k = i / inUnits;

	if (i < connUnits)
	{
		weights[i] += learning * gradient[k] * input[j];
		errorSignal[j] += gradient[k] * weights[i];
	}
}

//===========================================================================//

void DeviceLayer::feedbackward(const float* signal, float learning)
{
	// Inicializa o sinal funcional
	cudaMemset(errorSignal, 0, (inUnits - 1) * sizeof(float));

	// Calcula o gradiente
	feedbackwardDerivate<<<outBlocks, TPB>>>(signal, funcSignal, outUnits,
			gradient, bias);

	// Realiza a atualização dos pesos e cálculo do sinal de erro
	feedbackwardSum<<<connBlocks, TPB>>>(input, gradient, learning, inUnits,
			connUnits, weights, errorSignal);
}

//===========================================================================//

}

