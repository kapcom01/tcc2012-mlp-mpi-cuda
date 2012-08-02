#include "mlp/layer/OutputLayer.h"

namespace MLP
{

//===========================================================================//

OutputLayer::OutputLayer(uint inUnits, uint outUnits,
		const ActivationFunction* activation, LearningRate* learningRate)
		: Layer(inUnits, outUnits, activation, learningRate)
{
	this->learningRate = learningRate;
}

//===========================================================================//

OutputLayer::~OutputLayer()
{

}

//===========================================================================//

double OutputLayer::calculateError(uint i, const double* signal)
{
	return activation->derivate(nonActivatedOutput[i])
			* (signal[i] - activatedOutput[i]);
}

//===========================================================================//

void OutputLayer::updateLearningRate(const double* expectedOutput)
{
	// Ajusta a taxa de aprendizado
	learningRate->adjust(error, expectedOutput, outUnits);
}

//===========================================================================//

}

