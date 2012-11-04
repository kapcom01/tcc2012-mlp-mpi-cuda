#include "mlp/serial/HostLayer.h"

namespace ParallelMLP
{

//===========================================================================//

HostLayer::HostLayer(uint inUnits, uint outUnits)
	: Layer(inUnits, outUnits)
{
	// Aloca espaço para os vetores
	weights = new float[connUnits];
	bias = new float[outUnits];
	gradient = new float[outUnits];
	funcSignal = new float[outUnits];
	errorSignal = new float[inUnits];
}

//===========================================================================//

HostLayer::~HostLayer()
{
	delete[] weights;
	delete[] bias;
	delete[] gradient;
	delete[] funcSignal;
	delete[] errorSignal;
}

//===========================================================================//

void HostLayer::randomize()
{
	for (uint i = 0; i < connUnits; i++)
		weights[i] = random();

	for (uint i = 0; i < outUnits; i++)
		bias[i] = random();
}

//===========================================================================//

void HostLayer::feedforward(const float* input)
{
	this->input = input;

	// Inicializa o sinal funcional
	memset(funcSignal, 0, outUnits * sizeof(float));

	// Calcula o sinal funcional
	for (uint i = 0; i < connUnits; i++)
	{
		uint j = i % inUnits;
		uint k = i / inUnits;
		funcSignal[k] += weights[i] * input[j];
	}

	// Ativa o sinal funcional
	for (uint i = 0; i < outUnits; i++)
		funcSignal[i] = ACTIVATE(bias[i] + funcSignal[i]);
}

//===========================================================================//

void HostLayer::feedbackward(const float* signal, float learning)
{
	// Inicializa o sinal funcional
	memset(errorSignal, 0, inUnits * sizeof(float));

	// Calcula o gradiente
	for (uint i = 0; i < outUnits; i++)
	{
		gradient[i] = DERIVATE(funcSignal[i]) * signal[i];
		bias[i] += gradient[i];
	}

	// Atualiza os pesos e calcula o sinal de erro
	for (uint i = 0; i < connUnits; i++)
	{
		uint j = i % inUnits;
		uint k = i / inUnits;
		weights[i] += learning * gradient[k] * input[j];
		errorSignal[j] += gradient[k] * weights[i];
	}
}

//===========================================================================//

float HostLayer::random() const
{
	float r = rand() / (float) RAND_MAX;
	return 2 * r - 1;
}

//===========================================================================//

}

