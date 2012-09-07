#include "mlp/ExampleSet.h"

namespace MLP
{

//===========================================================================//

ExampleSet::ExampleSet(int relationID, SetType type)
{
	this->relationID = relationID;
	this->type = type;

	isNormalized = false;
	learning = 0.1;
	tolerance = 0.01;
	maxEpochs = 100000;
	error = 0;
	successRate = 0;
}

//===========================================================================//

ExampleSet::~ExampleSet()
{

}

//===========================================================================//

void ExampleSet::normalize()
{
	if (isNormalized)
		return;

	// Para cada instância
	for (uint k = 0; k < size(); k++)
	{
		// Normaliza cada coluna de entrada
		for (uint i = 0; i < inVars(); i++)
			adjust(input[k][i], stat[i].from, stat[i].to);

		// Normaliza cada coluna de saída
		for (uint t = 0; t < outVars(); t++)
		{
			uint i = inVars() + t;
			adjust(target[k][t], stat[i].from, stat[i].to);
		}
	}

	isNormalized = true;
}

//===========================================================================//

void ExampleSet::unnormalize()
{
	if (!isNormalized)
		return;

	// Para cada instância
	for (uint k = 0; k < size(); k++)
	{
		// Para cada coluna de entrada
		for (uint i = 0; i < inVars(); i++)
			adjust(input[k][i], stat[i].to, stat[i].from);

		// Para cada coluna de saída
		for (uint t = 0; t < outVars(); t++)
		{
			uint i = inVars() + t;
			adjust(target[k][t], stat[i].to, stat[i].from);
			adjust(output[k][t], stat[i].to, stat[i].from);
		}
	}

	isNormalized = false;
}

//===========================================================================//

void ExampleSet::adjust(double &x, const Range &from, const Range &to)
{
	x = (to.upper - to.lower) / (from.upper - from.lower)
			* (x - from.lower) + to.lower;
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

const vdouble& ExampleSet::getInput(uint i) const
{
	return input[i];
}

//===========================================================================//

const vdouble& ExampleSet::getTarget(uint i) const
{
	return target[i];
}

//===========================================================================//

}
