#include "mlp/serial/HostLayer.h"

namespace ParallelMLP
{

float random();

float activate(float x);

float derivate(float y);

//===========================================================================//

HostLayer::HostLayer(uint inUnits, uint outUnits)
	: Layer(inUnits, outUnits)
{

}

//===========================================================================//

HostLayer::~HostLayer()
{

}

//===========================================================================//

void HostLayer::randomize()
{
	for (uint n = 0; n < outUnits; n++)
		for (uint i = 0; i <= inUnits; i++)
			weights[n * inUnits + i] = random();
}

//===========================================================================//

void HostLayer::feedforward(const vec_float input)
{
	this->input = input;
	vec_float w(weights, inUnits);

	// Inicializa o sinal funcional
	thrust::fill(funcSignal.begin(), funcSignal.end(), 0);

	// Executa a ativação de cada neurônio
	for (uint n = 0; n < outUnits; n++)
	{
		// Calcula o sinal funcional
		for (uint i = 0; i < inUnits; i++)
			funcSignal[n] += input[i] * w(n)[i];
		funcSignal[n] += w(n)[inUnits];

		// Ativa a saída
		funcSignal[n] = activate(funcSignal[n]);
	}
}

//===========================================================================//

void HostLayer::feedback(const vec_float signal, float learning)
{
	vec_float w(weights, inUnits);

	// Inicializa o sinal funcional
	thrust::fill(errorSignal.begin(), errorSignal.end(), 0);

	// Atualiza os pesos de cada neurônio
	for (uint n = 0; n < outUnits; n++)
	{
		// Calcula o gradiente
		gradient[n] = derivate(funcSignal[n]) * signal[n];

		// Atualiza o peso e calcula o sinal de erro
		for (uint i = 0; i < inUnits; i++)
		{
			w(n)[i] += learning * gradient[n] * input[i];
			errorSignal[i] += gradient[n] * w(n)[i];
		}
		w(n)[inUnits] += learning * gradient[n];
	}
}

//===========================================================================//

float random()
{
	float r = rand() / (float) RAND_MAX;
	return 2 * r - 1;
}

//===========================================================================//

float activate(float x)
{
	return tanh(x);
}

//===========================================================================//

float derivate(float y)
{
	return (1 - y) * (1 + y);
}

//===========================================================================//

}

