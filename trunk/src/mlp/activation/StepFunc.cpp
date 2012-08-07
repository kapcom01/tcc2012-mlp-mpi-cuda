#include "mlp/activation/StepFunc.h"

namespace MLP
{

//===========================================================================//

StepFunc::StepFunc()
{

}

//===========================================================================//

StepFunc::~StepFunc()
{

}

//===========================================================================//

double StepFunc::activate(double x) const
{
	return (x >= 0) ? 1 : 0;
}

//===========================================================================//

double StepFunc::derivate(double x) const
{
	return 1;
}

//===========================================================================//

double StepFunc::initialValue(uint inUnits, uint outUnits) const
{
//	double max = 1 / (double) (inUnits + outUnits);
	double max = 1 / sqrt(inUnits);
	return randomBetween(-max, max);
}

}
