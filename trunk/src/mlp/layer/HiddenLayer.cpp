#include "mlp/layer/HiddenLayer.h"

namespace MLP
{

//===========================================================================//

HiddenLayer::HiddenLayer(uint inUnits, uint outUnits,
		const ActivationFunction* activation, const LearningRate* learningRate)
		: Layer(inUnits, outUnits, activation, learningRate)
{

}

//===========================================================================//

HiddenLayer::~HiddenLayer()
{

}

//===========================================================================//

double HiddenLayer::calculateError(uint i, const double* signal)
{
	return activation->derivate(nonActivatedOutput[i]) * signal[i];
}

//===========================================================================//

}
