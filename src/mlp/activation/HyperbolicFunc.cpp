#include "mlp/activation/HyperbolicFunc.h"

namespace MLP
{

//===========================================================================//

HyperbolicFunc::HyperbolicFunc()
{

}

//===========================================================================//

HyperbolicFunc::~HyperbolicFunc()
{

}

//===========================================================================//

double HyperbolicFunc::activate(double x) const
{
	return tanh(x);
}

//===========================================================================//

double HyperbolicFunc::derivate(double x) const
{
	double aux = 2 * cosh(x) / (cosh(2 * x) + 1);
	return aux * aux;
}

//===========================================================================//

double HyperbolicFunc::initialValue(uint inUnits, uint outUnits) const
{
//	double max = sqrt(6 / (double) (inUnits + outUnits));
	double max = 1 / sqrt(inUnits);
	return randomBetween(-max, max);
}

//===========================================================================//

}
