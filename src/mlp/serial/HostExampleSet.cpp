#include "mlp/serial/HostExampleSet.h"

namespace ParallelMLP
{

//===========================================================================//

HostExampleSet::HostExampleSet(const Relation &relation)
{
	size = relation.getNInstances();
	inVars = 1;
	outVars = 0;

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

	// Aloca espaço para as entradas
	input = new float[size * step];
	output = new float[size * outVars];
	stat = new Stat[step];

	setRelation(relation);
}

//===========================================================================//

HostExampleSet::~HostExampleSet()
{
	delete[] input;
	delete[] output;
	delete[] stat;
}

//===========================================================================//

void HostExampleSet::setRelation(const Relation& relation)
{
	inputIdx = outputIdx = statIdx = 0;

	// Para cada instância
	for (const Instance* inst : relation.getData())
	{
		uint i = 0;

		// Para cada valor da instância
		for (const Value* val : *inst)
		{
			// Adiciona a variável bias antes da variável de saída
			if (val->isLast())
				addBias();

			// Se for numérico
			if (val->getType() == NUMERIC)
				addValue(val->getNumber(), val->isLast());
			// Se for nominal
			else
				addValue(val->getNominal(),
						relation.getAttribute(i).getNominalCard(), val->isLast());

			i++;
		}
	}

	uint i = 0;

	// Para cada atributo
	for (const Attribute* attr : relation.getAttributes())
	{
		// Adiciona estatística para o bias
		if (attr->isLast())
			addStat(-1, 1, -1, 1, false);

		// Se for numérico
		if (attr->getType() == NUMERIC)
		{
			float min = BIG_M, max = -BIG_M;

			// Para cada instância
			for (const Instance* inst : relation.getData())
			{
				float val = inst->at(i)->getNumber();

				if (val > max)
					max = val;
				if (val < min)
					min = val;
			}

			addStat(min, max, -1, 1, attr->isLast());
		}
		// Se for nominal
		else
			addStat(-1, 1, attr->getNominalCard(), attr->isLast());

		i++;
	}
}

//===========================================================================//

void HostExampleSet::addBias()
{
	addValue(1, false);
}

//===========================================================================//

void HostExampleSet::addValue(float value, bool isTarget)
{
	input[inputIdx++] = value;
}

//===========================================================================//

void HostExampleSet::addValue(int value, uint card, bool isTarget)
{
	// Adiciona uma variável para cada possível valor
	for (uint i = 0; i < card; i++)
		if (i + 1 == value)
			addValue(1, isTarget);
		else
			addValue(0, isTarget);
}

//===========================================================================//

void HostExampleSet::addStat(float min, float max, float lower, float upper,
		bool isTarget)
{
	stat[statIdx++] = { {min, max}, {lower, upper} };
}

//===========================================================================//

void HostExampleSet::addStat(float lower, float upper, uint card, bool isTarget)
{
	// Adiciona uma variável para cada possível valor
	for (uint i = 0; i < card; i++)
		addStat(0, 1, lower, upper, isTarget);
}

//===========================================================================//

void HostExampleSet::normalize()
{
	if (isNormalized)
		return;

	// Normaliza cada entrada
	for (uint i = 0; i < size * step; i++)
	{
		uint j = i % step;
		adjust(input[i], stat[j].from, stat[j].to);
	}

	isNormalized = true;
}

//===========================================================================//

void HostExampleSet::unnormalize()
{
	if (!isNormalized)
		return;

	// Desnormaliza cada entrada
	for (uint i = 0; i < size * step; i++)
	{
		uint j = i % step;
		adjust(input[i], stat[j].to, stat[j].from);
	}

	// Desnormaliza cada saída da rede neural
	for (uint i = 0; i < size * outVars; i++)
	{
		uint j = i % outVars + inVars;
		adjust(output[i], stat[j].to, stat[j].from);
	}

	isNormalized = false;
}

//===========================================================================//

void HostExampleSet::adjust(float &x, const Range &from, const Range &to) const
{
	x = (to.upper - to.lower) / (from.upper - from.lower)
			* (x - from.lower) + to.lower;
}

//===========================================================================//

uint HostExampleSet::getInVars() const
{
	return inVars;
}

//===========================================================================//

uint HostExampleSet::getOutVars() const
{
	return outVars;
}

//===========================================================================//

uint HostExampleSet::getSize() const
{
	return size;
}

//===========================================================================//

const float* HostExampleSet::getInput() const
{
	return input;
}

//===========================================================================//

const float* HostExampleSet::getInput(uint i) const
{
	return &input[i * step];
}

//===========================================================================//

const float* HostExampleSet::getTarget(uint i) const
{
	return &input[i * step + inVars];
}

//===========================================================================//

void HostExampleSet::setOutput(uint i, float* output)
{
	float* inst = &(this->output[i * outVars]);
	memcpy(inst, output, outVars * sizeof(float));
}

//===========================================================================//

const Stat* HostExampleSet::getStat() const
{
	return stat;
}

//===========================================================================//

float HostExampleSet::getLearning() const
{
	return learning;
}

//===========================================================================//

void HostExampleSet::setLearning(float learning)
{
	this->learning = learning;
}

//===========================================================================//

float HostExampleSet::getTolerance() const
{
	return tolerance;
}

//===========================================================================//

void HostExampleSet::setTolerance(float tolerance)
{
	this->tolerance = tolerance;
}

//===========================================================================//

uint HostExampleSet::getMaxEpochs() const
{
	return maxEpochs;
}

//===========================================================================//

void HostExampleSet::setMaxEpochs(uint maxEpochs)
{
	this->maxEpochs = maxEpochs;
}

//===========================================================================//

float HostExampleSet::getError() const
{
	return error;
}

//===========================================================================//

void HostExampleSet::setError(float error)
{
	this->error = error;
}

//===========================================================================//

uint HostExampleSet::getEpochs() const
{
	return epochs;
}

//===========================================================================//

void HostExampleSet::setEpochs(uint epochs)
{
	this->epochs = epochs;
}

//===========================================================================//

float HostExampleSet::getTime() const
{
	return time;
}

//===========================================================================//

void HostExampleSet::setTime(float time)
{
	this->time = time;
}

//===========================================================================//

}
