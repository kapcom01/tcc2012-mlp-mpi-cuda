#include "mlp/activation/LogisticFunc.h"

namespace MLP
{

//===========================================================================//

LogisticFunc::LogisticFunc()
{

}

//===========================================================================//

LogisticFunc::~LogisticFunc()
{

}

//===========================================================================//

double LogisticFunc::activate(double x) const
{
	return 1 / (1 + exp(-x));
}

//===========================================================================//

double LogisticFunc::derivate(double x) const
{
	double aux = exp(x);
	return aux / ((aux + 1) * (aux + 1));
}

//===========================================================================//

double LogisticFunc::initialValue(uint inUnits, uint outUnits) const
{
//	double max = 4 * sqrt(6 / (double) (inUnits + outUnits));
	double max = 1 / sqrt(inUnits);
	return randomBetween(-max, max);
}

}
