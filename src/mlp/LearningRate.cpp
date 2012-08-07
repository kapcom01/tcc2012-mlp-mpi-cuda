#include "mlp/LearningRate.h"

namespace MLP
{

//===========================================================================//

LearningRate::LearningRate(double initialValue, uint searchTime)
{
	this->initial = initialValue;
	this->searchTime = searchTime;
	this->learningRate = initialValue;
}

//===========================================================================//

LearningRate::~LearningRate()
{

}

//===========================================================================//

double LearningRate::get() const
{
	return learningRate;
}

//===========================================================================//

double LearningRate::operator *() const
{
	return get();
}

//===========================================================================//

void LearningRate::adjust(uint iteration)
{
	learningRate = initial / (double) (1 + iteration / (double) searchTime);
}

//===========================================================================//

}
