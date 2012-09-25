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
	epochs = 0;
}

//===========================================================================//

ExampleSet::~ExampleSet()
{

}

//===========================================================================//

void ExampleSet::addValue(float value, bool isTarget)
{
	// Se for saída
	if (isTarget)
	{
		target.push_back(value);
		output.push_back(0);
	}
	// Se for entrada
	else
		input.push_back(value);
}

//===========================================================================//

void ExampleSet::addValue(int value, uint card, bool isTarget)
{
	// Adiciona uma variável para cada possível valor
	for (uint i = 0; i < card; i++)
		if (i + 1 == value)
			addValue(1, isTarget);
		else
			addValue(0, isTarget);
}

//===========================================================================//

void ExampleSet::addStat(float min, float max, float lower, float upper,
		bool isTarget)
{
	// Se for saída
	if (isTarget)
		outStat.push_back({ {min, max}, {lower, upper} });
	// Se for entrada
	else
		inStat.push_back({ {min, max}, {lower, upper} });
}

//===========================================================================//

void ExampleSet::addStat(float lower, float upper, uint card, bool isTarget)
{
	// Adiciona uma variável para cada possível valor
	for (uint i = 0; i < card; i++)
		addStat(0, 1, lower, upper, isTarget);
}

//===========================================================================//

bool ExampleSet::isTraining() const
{
	return (type == TRAINING);
}

//===========================================================================//

bool ExampleSet::isValidation() const
{
	return (type == VALIDATION);
}

//===========================================================================//

bool ExampleSet::isTest() const
{
	return (type == TEST);
}

//===========================================================================//

int ExampleSet::getID() const
{
	return relationID;
}

//===========================================================================//

int ExampleSet::getMLPID() const
{
	return mlpID;
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

void ExampleSet::setVars()
{
	this->inVars = input.size() / size;
	this->outVars = target.size() / size;
}

//===========================================================================//

uint ExampleSet::getSize() const
{
	return size;
}

//===========================================================================//

void ExampleSet::setSize(uint size)
{
	this->size = size;
}

//===========================================================================//

vec_float ExampleSet::getInput(uint i)
{
	return vec_float(input, inVars, i);
}

//===========================================================================//

vec_float ExampleSet::getTarget(uint i)
{
	return vec_float(target, outVars, i);
}

//===========================================================================//

float ExampleSet::getNumericOutput(uint k) const
{
	return output[k * outVars];
}

//===========================================================================//

int ExampleSet::getNominalOutput(uint k) const
{
	float max = output[k * outVars];
	uint ind = 0;

	// Percorre o vetor para encontrar o maior elemento
	for (uint i = k * outVars + 1; i < (k + 1) * outVars; i++)
		if (output[i] > max)
		{
			max = output[i];
			ind = i;
		}

	return ind % outVars + 1;
}

//===========================================================================//

void ExampleSet::setOutput(uint i, const vec_float output)
{
	vec_float this_out(this->output, outVars, i);
	memcpy(this_out.data, output.data, outVars * sizeof(float));
}

//===========================================================================//

int ExampleSet::getType() const
{
	return type;
}

//===========================================================================//

float ExampleSet::getLearning() const
{
	return learning;
}

//===========================================================================//

void ExampleSet::setLearning(float learning)
{
	this->learning = learning;
}

//===========================================================================//

float ExampleSet::getTolerance() const
{
	return tolerance;
}

//===========================================================================//

void ExampleSet::setTolerance(float tolerance)
{
	this->tolerance = tolerance;
}

//===========================================================================//

uint ExampleSet::getMaxEpochs() const
{
	return maxEpochs;
}

//===========================================================================//

void ExampleSet::setMaxEpochs(uint maxEpochs)
{
	this->maxEpochs = maxEpochs;
}

//===========================================================================//

float ExampleSet::getError() const
{
	return error;
}

//===========================================================================//

void ExampleSet::setError(float error)
{
	this->error = error;
}

//===========================================================================//

uint ExampleSet::getEpochs() const
{
	return epochs;
}

//===========================================================================//

void ExampleSet::setEpochs(uint epochs)
{
	this->epochs = epochs;
}

//===========================================================================//

float ExampleSet::getTime() const
{
	return time;
}

//===========================================================================//

void ExampleSet::setTime(float time)
{
	this->time = time;
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
