#include "mlp/InputSet.h"

namespace MLP
{

//===========================================================================//

InputSet::InputSet()
{

}

//===========================================================================//

InputSet::~InputSet()
{

}

//===========================================================================//

uint InputSet::inVars()
{
	return input[0].size();
}

//===========================================================================//

uint InputSet::outVars()
{
	return target[0].size();
}

//===========================================================================//

uint InputSet::size()
{
	return input.size();
}

//===========================================================================//

void InputSet::pushInstance()
{
	input.push_back(vector<double>());
	target.push_back(vector<double>());
}

//===========================================================================//

void InputSet::addValue(const double &value, bool isTarget)
{
	// Seleciona o vetor correto
	vector<double> &values = (isTarget) ? target.back() : input.back();

	// Adiciona o valor numérico
	values.push_back(value);
}

//===========================================================================//

void InputSet::addValue(const int &value, const uint &card, bool isTarget)
{
	// Seleciona o vetor correto
	vector<double> &values = (isTarget) ? target.back() : input.back();

	// Adiciona uma variável para cada possível valor
	for (uint i = 0; i < card; i++)
		if (i + 1 == value)
			values.push_back(1);
		else
			values.push_back(0);
}

//===========================================================================//

}
