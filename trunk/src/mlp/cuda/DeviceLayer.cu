#include "mlp/cuda/DeviceLayer.h"

namespace ParallelMLP
{

__device__
float d_random(curandState* state);

__host__ __device__
float d_activate(float x);

__host__ __device__
float d_derivate(float y);

//===========================================================================//

DeviceLayer::DeviceLayer()
{

}

//===========================================================================//

DeviceLayer::DeviceLayer(uint inUnits, uint outUnits)
{
	init(inUnits, outUnits);
}

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

void DeviceLayer::init(uint inUnits, uint outUnits)
{
	Layer::init(inUnits, outUnits);

	devGradient.resize(outUnits);
	devFuncSignal.resize(outUnits);
	devErrorSignal.resize(inUnits);
	devState.resize(outUnits * (inUnits + 1));

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

	weights(n)[i] = d_random(&state(n)[i]);
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
	funcSignal[n] = d_activate(funcSignal[n]);
}

//===========================================================================//

void DeviceLayer::feedforward(const vec_float &input)
{
	this->input = input;

	// Inicializa o sinal funcional
	rawFuncSignal.deviceClear();

	// Calcula as somas ponderadas das entradas
	feedforwardSum<<<outUnits, inUnits>>>(rawFuncSignal, rawWeights, input);

	// Ativa as saídas de cada neurônio
	feedforwardActivate<<<outUnits, 1>>>(rawFuncSignal, rawWeights, inUnits);

	cout << "       |-> Input: ";
	vec_float aux = input;
	device_ptr<float> ptr = thrust::device_pointer_cast(aux.data());
	dv_float in(ptr, ptr + inUnits);
	for (uint i = 0; i < inUnits; i++)
		cout << in[i] << " ";
	cout << endl;

	cout << "       |-> Output: ";
	for (uint i = 0; i < outUnits; i++)
		cout << devFuncSignal[i] << " ";
	cout << endl;
}

//===========================================================================//

__global__
void feedbackDerivate(vec_float funcSignal, vec_float weights,
		vec_float gradient, vec_float signal, uint inUnits, float learning)
{
	int n = blockIdx.x;

	gradient[n] = d_derivate(funcSignal[n]) * signal[n];
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

void DeviceLayer::feedback(const vec_float &signal, float learning)
{
	// Inicializa o sinal funcional
	rawErrorSignal.deviceClear();

	// Calcula o gradiente
	feedbackDerivate<<<outUnits, 1>>>(rawFuncSignal, rawWeights, rawGradient,
			signal, inUnits, learning);

	// Realiza a atualização dos pesos e cálculo do sinal de erro
	feedbackSum<<<outUnits, inUnits>>>(rawErrorSignal, rawWeights, rawGradient,
			input, learning);

	cout << "       |-> Signal: ";
	vec_float aux = signal;
	device_ptr<float> ptr = thrust::device_pointer_cast(aux.data());
	dv_float sig(ptr, ptr + inUnits);
	for (uint i = 0; i < outUnits; i++)
		cout << sig[i] << " ";
	cout << endl;

	cout << "       |-> Error: ";
	for (uint i = 0; i < inUnits; i++)
		cout << devErrorSignal[i] << " ";
	cout << endl;
}

//===========================================================================//

__device__
float d_random(curandState* state)
{
	float r = curand(state);
	return 2 * r - 1;
}

//===========================================================================//

__host__ __device__
float d_activate(float x)
{
	return tanh(x);
}

//===========================================================================//

__host__ __device__
float d_derivate(float y)
{
	return (1 - y) * (1 + y);
}

//===========================================================================//

}

