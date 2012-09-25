#include "mlp/common/Neuron.h"

namespace ParallelMLP
{

//===========================================================================//

Neuron::Neuron(uint inUnits)
{
	this->inUnits = inUnits;
	weights.resize(inUnits + 1);
	gradient = 0;
}

//===========================================================================//

Neuron::~Neuron()
{

}

//===========================================================================//

float Neuron::getWeight(uint i)
{
	return weights[i];
}

//===========================================================================//

void Neuron::setWeight(uint i, float weight)
{
	weights[i] = weight;
}

//===========================================================================//

}
