#include "mlp/common/OutLayer.h"

namespace ParallelMLP
{

//===========================================================================//

OutLayer::OutLayer(uint inUnits, uint outUnits)
	: Layer(inUnits, outUnits)
{
	samples = totalError = 0;
	error = NULL;
}

//===========================================================================//

OutLayer::~OutLayer()
{

}

//===========================================================================//

void OutLayer::clearError()
{
	totalError = 0;
	samples = 0;
}

//===========================================================================//

void OutLayer::incError(float inc)
{
	// Calcula o erro quadrático médio
	totalError = (totalError * samples + inc) / (float) (samples + outUnits);
	samples += outUnits;
}

//===========================================================================//

float OutLayer::getError()
{
	return totalError;
}

//===========================================================================//

}

