#include "mlp/activation/LogisticFunction.h"

namespace MLP
{

//===========================================================================//

LogisticFunction::LogisticFunction()
{

}

//===========================================================================//

LogisticFunction::~LogisticFunction()
{

}

//===========================================================================//

double LogisticFunction::activate(double x) const
{
	return 1 / (1 + exp(-x));
}

//===========================================================================//

double LogisticFunction::derivate(double x) const
{
	double aux = exp(x);
	return aux / ((aux + 1) * (aux + 1));
}

//===========================================================================//

double LogisticFunction::initialValue(uint inUnits, uint outUnits) const
{
	double max = 4 * sqrt(6 / (double) (inUnits + outUnits));
	return randomBetween(-max, max);
}

}
