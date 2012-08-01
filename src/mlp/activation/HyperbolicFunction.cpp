#include "mlp/activation/HyperbolicFunction.h"

namespace MLP
{

//===========================================================================//

HyperbolicFunction::HyperbolicFunction()
{

}

//===========================================================================//

HyperbolicFunction::~HyperbolicFunction()
{

}

//===========================================================================//

double HyperbolicFunction::activate(double x) const
{
	return tanh(x);
}

//===========================================================================//

double HyperbolicFunction::derivate(double x) const
{
	double aux = 2 * cosh(x) / (cosh(2 * x) + 1);
	return aux * aux;
}

//===========================================================================//

double HyperbolicFunction::initialValue(uint inUnits, uint outUnits) const
{
	double max = sqrt(6 / (double) (inUnits + outUnits));
	return randomBetween(-max, max);
}

//===========================================================================//

}
