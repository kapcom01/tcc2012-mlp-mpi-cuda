#include "mlp/common/Layer.h"

namespace ParallelMLP
{

//===========================================================================//

Layer::Layer(uint inUnits, uint outUnits)
	: weights(outUnits * (inUnits + 1))
{
	this->inUnits = inUnits;
	this->outUnits = outUnits;
}

//===========================================================================//

Layer::~Layer()
{

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

vec_float Layer::getFuncSignal()
{
	return rawFuncSignal;
}

//===========================================================================//

vec_float Layer::getErrorSignal()
{
	return rawErrorSignal;
}

//===========================================================================//

}

