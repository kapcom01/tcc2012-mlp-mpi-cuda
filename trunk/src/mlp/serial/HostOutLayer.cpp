#include "mlp/serial/HostOutLayer.h"

namespace ParallelMLP
{

//===========================================================================//

HostOutLayer::HostOutLayer(uint inUnits, uint outUnits)
	: HostLayer(inUnits, outUnits)
{
	error = new float[outUnits];
	samples = 0;
	totalError = 0;
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

	// Calcula o erro quadrático médio
	totalError = (totalError * samples + sum) / (samples + outUnits);
	samples += outUnits;
}

//===========================================================================//

void HostOutLayer::feedbackward(const float* target, float learning)
{
	// Calcula o error e chama a função de feedbackward do pai
	calculateError(target);
	HostLayer::feedbackward(error, learning);
}

//===========================================================================//

void HostOutLayer::clearError()
{
	totalError = 0;
	samples = 0;
}

//===========================================================================//

float HostOutLayer::getError()
{
	return totalError;
}

//===========================================================================//

}

