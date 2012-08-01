#include "mlp/LearningRate.h"

namespace MLP
{

//===========================================================================//

LearningRate::LearningRate(double initialValue, double min, double max)
{
	this->min = min;
	this->max = max;
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

void LearningRate::adjust(const double* error, const double* expectedOutput, uint size)
{
	double sum = 0;

	// Calcula a soma dos erros percentuais em relação à saída esperada
	for (uint i = 0; i < size; i++)
		sum += fabs(error[i]) / expectedOutput[i];

	// Calcula o erro médio
	double avg = sum / (double) size;

	// Altera a taxa de aprendizado
	learningRate = (max - min) * (1 - exp(-avg)) + min;
}

//===========================================================================//

}
