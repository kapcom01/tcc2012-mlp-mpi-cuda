#include "mlp/cuda/DeviceLayer.h"

namespace ParallelMLP
{

__device__
float random(curandState* state);

__host__ __device__
float activate(float x);

__host__ __device__
float derivate(float y);

//===========================================================================//

__global__
void initRandState(vec_rand state, int seed)
{
	int n = blockIdx.x;
	int i = threadIdx.x;
	int id = n * blockDim.x + i;

	curand_init(seed + id, 0, 0, &state(n)[i]);
}

//===========================================================================//

DeviceLayer::DeviceLayer(uint inUnits, uint outUnits)
	: Layer(inUnits, outUnits), devGradient(outUnits), devFuncSignal(outUnits),
	  devErrorSignal(inUnits), devState(outUnits * (inUnits + 1))
{
	rawGradient = vec_float(devGradient);
	rawFuncSignal = vec_float(devFuncSignal);
	rawErrorSignal = vec_float(devErrorSignal);
	rawState = vec_rand(devState, inUnits + 1);

	initRandState<<<outUnits, inUnits + 1>>>(rawState, rand());
}

//===========================================================================//

DeviceLayer::~DeviceLayer()
{

}

//===========================================================================//

void DeviceLayer::copyToDevice()
{
	// Copia os pesos para a GPU
	devWeights = weights;

	// Atribui o vetor puro
	rawWeights = vec_float(devWeights, inUnits + 1);
}

//===========================================================================//

void DeviceLayer::copyToHost()
{
	// Copia os pesos para a CPU
	weights = devWeights;
}

//===========================================================================//

__global__
void randomizeWeight(vec_float weights, vec_rand state)
{
	int n = blockIdx.x;
	int i = threadIdx.x;

	weights(n)[i] = random(&state(n)[i]);
}

//===========================================================================//

void DeviceLayer::randomize()
{
	// Copia para a GPU
	copyToDevice();

	// Randomiza os pesos
	randomizeWeight<<<outUnits, inUnits + 1>>>(rawWeights, rawState);

	// Copia para  CPU
	copyToHost();
}

//===========================================================================//

void DeviceLayer::initOperation()
{
	copyToDevice();
}

//===========================================================================//

void DeviceLayer::endOperation()
{
	copyToHost();
}

//===========================================================================//

__global__
void feedforwardSum(vec_float funcSignal, vec_float weights, vec_float input)
{
	int n = blockIdx.x;
	int i = threadIdx.x;

	atomicAdd(&funcSignal[n], input[i] * weights(n)[i]);
}

//===========================================================================//

__global__
void feedforwardActivate(vec_float funcSignal, vec_float weights, uint inUnits)
{
	int n = blockIdx.x;

	funcSignal[n] += weights(n)[inUnits];
	funcSignal[n] = activate(funcSignal[n]);
}

//===========================================================================//

void DeviceLayer::feedforward(const vec_float input)
{
	this->input = input;

	// Inicializa o sinal funcional
	rawFuncSignal.deviceClear();

	// Calcula as somas ponderadas das entradas
	feedforwardSum<<<outUnits, inUnits>>>(rawFuncSignal, rawWeights, input);

	// Ativa as saídas de cada neurônio
	feedforwardActivate<<<outUnits, 1>>>(rawFuncSignal, rawWeights, inUnits);
}

//===========================================================================//

__global__
void feedbackDerivate(vec_float funcSignal, vec_float weights,
		vec_float gradient, vec_float signal, uint inUnits, float learning)
{
	int n = blockIdx.x;

	gradient[n] = derivate(funcSignal[n]) * signal[n];
	weights(n)[inUnits] += learning * gradient[n];
}

//===========================================================================//

__global__
void feedbackSum(vec_float errorSignal, vec_float weights, vec_float gradient,
		vec_float input, float learning)
{
	int n = blockIdx.x;
	int i = threadIdx.x;

	weights(n)[i] += learning * gradient[n] * input[i];
	atomicAdd(&errorSignal[i], gradient[n] * weights(n)[i]);
}

//===========================================================================//

void DeviceLayer::feedback(const vec_float signal, float learning)
{
	// Inicializa o sinal funcional
	rawErrorSignal.deviceClear();

	// Calcula o gradiente
	feedbackDerivate<<<outUnits, 1>>>(rawFuncSignal, rawWeights, rawGradient,
			signal, inUnits, learning);

	// Realiza a atualização dos pesos e cálculo do sinal de erro
	feedbackSum<<<outUnits, inUnits>>>(rawErrorSignal, rawWeights, rawGradient,
			input, learning);
}

//===========================================================================//

__device__
float random(curandState* state)
{
	float r = curand(state);
	return 2 * r - 1;
}

//===========================================================================//

__host__ __device__
float activate(float x)
{
	return tanh(x);
}

//===========================================================================//

__host__ __device__
float derivate(float y)
{
	return (1 - y) * (1 + y);
}

//===========================================================================//

}

