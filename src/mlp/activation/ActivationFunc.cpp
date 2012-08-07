#include "mlp/activation/ActivationFunc.h"

namespace MLP
{

//===========================================================================//

ActivationFunc::ActivationFunc()
{

}

//===========================================================================//

ActivationFunc::~ActivationFunc()
{

}

//===========================================================================//

double ActivationFunc::randomBetween(double min, double max) const
{
	double r = rand() / (double) RAND_MAX;
	return r * (max - min) + min;
}

//===========================================================================//

}
