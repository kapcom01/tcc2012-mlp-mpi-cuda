#include "mlp/common/ExampleSet.h"

namespace ParallelMLP
{

//===========================================================================//

ExampleSet::ExampleSet(const Relation &relation)
{
	size = relation.getNInstances(), inVars = 1, outVars = 0;

	// Calcula a quantidade de variáveis de entrada e saída que existirão
	for (const Attribute* attr : relation.getAttributes())
	{
		uint inc = attr->isNumeric() ? 1 : attr->getNominalCard();

		if (attr->isLast())
			outVars += inc;
		else
			inVars += inc;
	}

	// Soma das variáveis de entrada, do bias e das variáveis de saída
	step = inVars + outVars;

	// Inicializa as variáveis
	input = output = NULL;
	stat = NULL;
	inputIdx = outputIdx = statIdx = 0;
	maxEpochs = epochs = 0;
	error = time = learning = tolerance = 0;
	isNormalized = false;
}

//===========================================================================//

ExampleSet::~ExampleSet()
{

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

const float* ExampleSet::getInput() const
{
	return input;
}

//===========================================================================//

const float* ExampleSet::getInput(uint i) const
{
	return &input[i * step];
}

//===========================================================================//

const float* ExampleSet::getTarget(uint i) const
{
	return &input[i * step + inVars];
}

//===========================================================================//

const Stat* ExampleSet::getStat() const
{
	return stat;
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

}
