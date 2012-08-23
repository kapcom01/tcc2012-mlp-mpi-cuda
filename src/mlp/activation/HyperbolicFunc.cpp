#include "mlp/activation/HyperbolicFunc.h"

namespace MLP
{

//===========================================================================//

HyperbolicFunc::HyperbolicFunc()
		: ActivationFunc(HYPERBOLIC)
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

double HyperbolicFunc::derivate(double y) const
{
	return (1 - y) * (1 + y);
}

//===========================================================================//

Range HyperbolicFunc::getRange() const
{
	return {-1, 1};
}

//===========================================================================//

}
