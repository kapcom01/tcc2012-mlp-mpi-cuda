#include "mlp/ExampleSet.h"

namespace MLP
{

//===========================================================================//

ExampleSet::ExampleSet()
{
	momentum = 0.9;
	learningRate = 0.1;
	minSuccessRate = 0.95;
	maxTolerance = 0.01;
	successRate = 0;
	maxEpochs = 100000;
}

//===========================================================================//

ExampleSet::~ExampleSet()
{

}

//===========================================================================//

uint ExampleSet::inVars() const
{
	return input[0].size();
}

//===========================================================================//

uint ExampleSet::outVars() const
{
	return target[0].size();
}

//===========================================================================//

uint ExampleSet::size() const
{
	return input.size();
}

//===========================================================================//

const vector<double>& ExampleSet::getInput(uint i) const
{
	return input[i];
}

//===========================================================================//

const vector<double>& ExampleSet::getTarget(uint i) const
{
	return target[i];
}

//===========================================================================//

void ExampleSet::pushInstance()
{
	input.push_back(vector<double>());
	target.push_back(vector<double>());
}

//===========================================================================//

void ExampleSet::addValue(const double &value, bool isTarget)
{
	// Seleciona o vetor correto
	vector<double> &values = (isTarget) ? target.back() : input.back();

	// Adiciona o valor numérico
	values.push_back(value);
}

//===========================================================================//

void ExampleSet::addValue(const int &value, const uint &card, bool isTarget)
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
