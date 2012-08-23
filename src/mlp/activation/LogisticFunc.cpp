#include "mlp/activation/LogisticFunc.h"

namespace MLP
{

//===========================================================================//

LogisticFunc::LogisticFunc()
		: ActivationFunc(LOGISTIC)
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

double LogisticFunc::derivate(double y) const
{
	return y * (1 - y);
}

//===========================================================================//

Range LogisticFunc::getRange() const
{
	return {0, 1};
}

//===========================================================================//

}
