#include "mlp/serial/HostOutLayer.h"

namespace ParallelMLP
{

//===========================================================================//

HostOutLayer::HostOutLayer(uint inUnits, uint outUnits)
	: Layer(inUnits, outUnits), OutLayer(inUnits, outUnits),
	  HostLayer(inUnits, outUnits)
{
	error = new float[outUnits];
}

//===========================================================================//

HostOutLayer::~HostOutLayer()
{
	delete[] error;
}

//===========================================================================//

void HostOutLayer::calculateError(const float* target)
{
	float sum = 0;

	// Calcula o erro cometido pela rede
	for (uint i = 0; i < outUnits; i++)
	{
		error[i] = target[i] - funcSignal[i];
		sum += error[i] * error[i];
	}

	// Incrementa o erro
	incError(sum);
}

//===========================================================================//

void HostOutLayer::feedforward(const float* input)
{
	return HostLayer::feedforward(input);
}

//===========================================================================//

void HostOutLayer::feedbackward(const float* target, float learning)
{
	// Calcula o error e chama a função de feedbackward do pai
	calculateError(target);
	HostLayer::feedbackward(error, learning);
}

//===========================================================================//

}

