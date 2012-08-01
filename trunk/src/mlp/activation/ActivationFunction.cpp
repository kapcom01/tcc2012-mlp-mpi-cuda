#include "mlp/activation/ActivationFunction.h"

namespace MLP
{

//===========================================================================//

ActivationFunction::ActivationFunction()
{

}

//===========================================================================//

ActivationFunction::~ActivationFunction()
{

}

//===========================================================================//

double ActivationFunction::randomBetween(double min, double max) const
{
	double r = rand() / (double) RAND_MAX;
	return r * (max - min) + min;
}

//===========================================================================//

}
