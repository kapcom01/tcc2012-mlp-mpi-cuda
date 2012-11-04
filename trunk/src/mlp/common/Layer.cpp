#include "mlp/common/Layer.h"

namespace ParallelMLP
{

//===========================================================================//

Layer::Layer(uint inUnits, uint outUnits)
{
	this->inUnits = inUnits;
	this->outUnits = outUnits;
	this->connUnits = inUnits * outUnits;

	funcSignal = errorSignal = gradient = weights = bias = NULL;
	input = NULL;
}

//===========================================================================//

Layer::~Layer()
{

}

//===========================================================================//

uint Layer::getInUnits()
{
	return inUnits;
}

//===========================================================================//

uint Layer::getOutUnits()
{
	return outUnits;
}

//===========================================================================//

float* Layer::getFuncSignal()
{
	return funcSignal;
}

//===========================================================================//

float* Layer::getErrorSignal()
{
	return errorSignal;
}

//===========================================================================//

}

