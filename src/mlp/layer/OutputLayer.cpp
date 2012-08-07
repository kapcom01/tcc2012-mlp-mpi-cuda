#include "mlp/layer/OutputLayer.h"

namespace MLP
{

//===========================================================================//

OutputLayer::OutputLayer(uint inUnits, uint outUnits,
		const ActivationFunc* activation)
		: Layer(inUnits, outUnits, activation)
{

}

//===========================================================================//

OutputLayer::~OutputLayer()
{

}

//===========================================================================//

double OutputLayer::calculateError(uint i, const double* signal)
{
	return activation->derivate(weightedSum[i])
			* (signal[i] - output[i]);
}

//===========================================================================//

}

