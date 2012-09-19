#include "mlp/common/ExampleSet.h"

namespace ParallelMLP
{

//===========================================================================//

ExampleSet::ExampleSet(int relationID, int mlpID, SetType type)
{
	this->relationID = relationID;
	this->mlpID = mlpID;
	this->type = type;

	isNormalized = false;
	learning = 0.1;
	tolerance = 0.01;
	maxEpochs = 100000;
	error = 0;
	time = 0;
	size = 0;
	inVars = 0;
	outVars = 0;
}

//===========================================================================//

ExampleSet::~ExampleSet()
{

}

//===========================================================================//

void ExampleSet::setProperties(float learning, uint maxEpochs,
		float tolerance)
{
	this->learning = learning;
	this->maxEpochs = maxEpochs;
	this->tolerance = tolerance;
}

//===========================================================================//

uint ExampleSet::getInVars() const
{
	return inVars;
}

//===========================================================================//

uint ExampleSet::getOutVars() const
{
	return outVars;
}

//===========================================================================//

uint ExampleSet::getSize() const
{
	return size;
}

//===========================================================================//

const hv_float ExampleSet::getInput(uint i) const
{
	hv_float vec(inVars);
	for (uint j = 0; j < inVars; j++)
		vec[j] = input[i * inVars + j];
	return vec;
}

//===========================================================================//

const hv_float ExampleSet::getTarget(uint i) const
{
	hv_float vec(outVars);
	for (uint j = 0; j < outVars; j++)
		vec[j] = target[i * outVars + j];
	return vec;
}

//===========================================================================//

void ExampleSet::setOutput(uint i, const hv_float &output)
{
	for (uint j = i * outVars; j < (i + 1) * outVars; j++)
		this->output[j] = output[i];
}

//===========================================================================//

void ExampleSet::print()
{
	for (uint k = 0; k < size; k++)
	{
		for (uint i = 0; i < inVars; i++)
			cout << input[k * inVars + i] << " ";
		cout << "| ";
		for (uint i = 0; i < outVars; i++)
			cout << target[k * outVars + i] << " ";
		cout << endl;
	}
}

}
