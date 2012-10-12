#include "mlp/common/OutLayer.h"

namespace ParallelMLP
{

//===========================================================================//

OutLayer::OutLayer()
{
	clearTotalError();
}

//===========================================================================//

OutLayer::OutLayer(uint inUnits, uint outUnits)
{
	init(inUnits, outUnits);
}

//===========================================================================//

void OutLayer::init(uint inUnits, uint outUnits)
{
	Layer::init(inUnits, outUnits);
	clearTotalError();
}

//===========================================================================//

OutLayer::~OutLayer()
{

}

//===========================================================================//

void OutLayer::clearTotalError()
{
	totalError = 0;
	samples = 0;
}

//===========================================================================//

void OutLayer::incTotalError(float value, uint weight)
{
	totalError = (samples * totalError + value) / (double) (samples + weight);
	samples += weight;
}

//===========================================================================//

float OutLayer::getTotalError()
{
	return totalError;
}

//===========================================================================//

}

