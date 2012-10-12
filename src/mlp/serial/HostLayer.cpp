#include "mlp/serial/HostLayer.h"

namespace ParallelMLP
{

float random();

float activate(float x);

float derivate(float y);

//===========================================================================//

HostLayer::HostLayer()
{

}

//===========================================================================//

HostLayer::HostLayer(uint inUnits, uint outUnits)
{
	init(inUnits, outUnits);
}

//===========================================================================//

void HostLayer::init(uint inUnits, uint outUnits)
{
	Layer::init(inUnits, outUnits);

	gradient.resize(outUnits);
	funcSignal.resize(outUnits);
	errorSignal.resize(inUnits);

	rawWeights = vec_float(weights, inUnits + 1);
	rawFuncSignal = vec_float(funcSignal);
	rawErrorSignal = vec_float(errorSignal);
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

void HostLayer::initOperation()
{

}

//===========================================================================//

void HostLayer::endOperation()
{

}

//===========================================================================//

void HostLayer::feedforward(const vec_float &input)
{
	this->input = input;

	// Inicializa o sinal funcional
	rawFuncSignal.hostClear();

	// Executa a ativação de cada neurônio
	for (uint n = 0; n < outUnits; n++)
	{
		// Calcula o sinal funcional
		for (uint i = 0; i < inUnits; i++)
			rawFuncSignal[n] += input[i] * rawWeights(n)[i];
		rawFuncSignal[n] += rawWeights(n)[inUnits];

		// Ativa a saída
		rawFuncSignal[n] = activate(rawFuncSignal[n]);
	}
}

//===========================================================================//

void HostLayer::feedback(const vec_float &signal, float learning)
{
	// Inicializa o sinal funcional
	rawErrorSignal.hostClear();

	// Atualiza os pesos de cada neurônio
	for (uint n = 0; n < outUnits; n++)
	{
		// Calcula o gradiente
		gradient[n] = derivate(funcSignal[n]) * signal[n];

		// Atualiza o peso e calcula o sinal de erro
		for (uint i = 0; i < inUnits; i++)
		{
			rawWeights(n)[i] += learning * gradient[n] * input[i];
			errorSignal[i] += gradient[n] * rawWeights(n)[i];
		}
		rawWeights(n)[inUnits] += learning * gradient[n];
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

