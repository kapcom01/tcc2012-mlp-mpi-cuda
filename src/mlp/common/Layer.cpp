#include "mlp/common/Layer.h"

namespace ParallelMLP
{

//===========================================================================//

Layer::Layer(uint inUnits, uint outUnits)
{
	this->inUnits = inUnits;
	this->outUnits = outUnits;

	// Aloca os vetores de saída, de feedback e de erro
	funcSignal.resize(outUnits);
	errorSignal.resize(inUnits);
}

//===========================================================================//

Layer::~Layer()
{
	for (uint i = 0; i < neurons.size(); i++)
		delete neurons[i];
}

//===========================================================================//

void Layer::randomize()
{
	for (uint n = 0; n < outUnits; n++)
		neurons[n]->randomize();
}

//===========================================================================//

vec_float Layer::getFuncSignal()
{
	return vec_float(funcSignal);
}

//===========================================================================//

vec_float Layer::getErrorSignal()
{
	return vec_float(errorSignal);
}

//===========================================================================//

uint Layer::getInUnits() const
{
	return inUnits;
}

//===========================================================================//

uint Layer::getOutUnits() const
{
	return outUnits;
}

//===========================================================================//

float Layer::getWeight(uint n, uint i) const
{
	return neurons[n]->getWeight(i);
}

//===========================================================================//

void Layer::setWeight(uint n, uint i, float weight)
{
	neurons[n]->setWeight(i, weight);
}

//===========================================================================//

}

