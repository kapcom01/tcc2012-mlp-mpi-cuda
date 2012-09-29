#include "mlp/common/Layer.h"

namespace ParallelMLP
{

//===========================================================================//

Layer::Layer(uint inUnits, uint outUnits)
	: weights(outUnits * (inUnits + 1)), gradient(outUnits),
	  funcSignal(outUnits), errorSignal(inUnits)
{
	this->inUnits = inUnits;
	this->outUnits = outUnits;
}

//===========================================================================//

Layer::~Layer()
{

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
	return weights[n * inUnits + i];
}

//===========================================================================//

void Layer::setWeight(uint n, uint i, float weight)
{
	weights[n * inUnits + i] = weight;
}

//===========================================================================//

}

