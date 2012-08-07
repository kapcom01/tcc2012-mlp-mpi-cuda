#include "mlp/layer/HiddenLayer.h"

namespace MLP
{

//===========================================================================//

HiddenLayer::HiddenLayer(uint inUnits, uint outUnits,
		const ActivationFunc* activation)
		: Layer(inUnits, outUnits, activation)
{

}

//===========================================================================//

HiddenLayer::~HiddenLayer()
{

}

//===========================================================================//

double HiddenLayer::calculateError(uint i, const double* signal)
{
	return activation->derivate(weightedSum[i]) * signal[i];
}

//===========================================================================//

}
