#include "mlp/activation/LinearFunc.h"

namespace MLP
{

//===========================================================================//

LinearFunc::LinearFunc()
{

}

//===========================================================================//

LinearFunc::~LinearFunc()
{

}

//===========================================================================//

double LinearFunc::activate(double x) const
{
	return x;
}

//===========================================================================//

double LinearFunc::derivate(double x) const
{
	return 1;
}

//===========================================================================//

double LinearFunc::initialValue(uint inUnits, uint outUnits) const
{
//	double max = 1 / (double) (inUnits + outUnits);
	double max = 1 / sqrt(inUnits);
	return randomBetween(-max, max);
}

}
