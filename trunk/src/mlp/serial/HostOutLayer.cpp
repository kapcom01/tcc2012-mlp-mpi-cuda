#include "mlp/serial/HostOutLayer.h"

namespace ParallelMLP
{

//===========================================================================//

HostOutLayer::HostOutLayer()
{

}

//===========================================================================//

HostOutLayer::HostOutLayer(uint inUnits, uint outUnits)
{
	init(inUnits, outUnits);
}

//===========================================================================//

void HostOutLayer::init(uint inUnits, uint outUnits)
{
	this->inUnits = inUnits;
	this->outUnits = outUnits;

	weights.resize(outUnits * (inUnits + 1));

	gradient.resize(outUnits);
	funcSignal.resize(outUnits);
	errorSignal.resize(inUnits);

	rawWeights = vec_float(weights, inUnits + 1);
	rawFuncSignal = vec_float(funcSignal);
	rawErrorSignal = vec_float(errorSignal);

	error.resize(outUnits);
	rawError = vec_float(error);
}

//===========================================================================//

HostOutLayer::~HostOutLayer()
{

}

//===========================================================================//

void HostOutLayer::calculateError(const vec_float &target)
{
	// Calcula o erro cometido pela rede
	for (uint i = 0; i < error.size(); i++)
	{
		error[i] = target[i] - funcSignal[i];
		incTotalError(error[i] * error[i]);
	}
}

//===========================================================================//

void HostOutLayer::feedback(const vec_float &target, float learning)
{
	// Calcula o error e chama a função de feedback do pai
	calculateError(target);
	HostLayer::feedback(rawError, learning);
}

//===========================================================================//

}

