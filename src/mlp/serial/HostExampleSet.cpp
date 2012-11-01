#include "mlp/serial/HostExampleSet.h"

namespace ParallelMLP
{

//===========================================================================//

HostExampleSet::HostExampleSet(uint size, uint inVars, uint outVars)
	: ExampleSet(size, inVars, outVars)
{
	init();
	randomize();
}

//===========================================================================//

HostExampleSet::HostExampleSet(const Relation &relation)
	: ExampleSet(relation)
{
	init();
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

void HostExampleSet::init()
{
	// Aloca espaço para as entradas
	input = new float[size * step];
	output = new float[size * outVars];
	stat = new Stat[step];
}

//===========================================================================//

void HostExampleSet::randomize()
{
	// Seta valores aleatórios (1 para os valores de bias)
	for (uint i = 0; i < size * step; i++)
		input[i] = (i % size == step - 1) ? 1 : rand() % 100;

	// Adiciona estatísticas
	addStatistics();
}

//===========================================================================//

void HostExampleSet::setRelation(const Relation& relation)
{
	inputIdx = outputIdx = 0;

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

	// Adiciona estatísticas
	addStatistics();
}

//===========================================================================//

void HostExampleSet::addStatistics()
{
	for (uint i = 0; i < step; i++)
	{
		float min = BIG_M, max = -BIG_M;

		// Recupera o menor e maior valor
		for (uint j = 0; j < size; j++)
		{
			float val = input[j * step + i];
			min = (val < min) ? val : min;
			max = (val > max) ? val : max;
		}

		// Estatísticas para o bias
		if (i == step - 1)
			stat[i] = { {-1, 1}, {-1, 1} };
		else
			stat[i] = { {min, max}, {-1, 1} };
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

void HostExampleSet::adjust(float &x, const Range &from, const Range &to)
		const
{
	x = (to.upper - to.lower) / (from.upper - from.lower)
			* (x - from.lower) + to.lower;
}

//===========================================================================//

void HostExampleSet::setOutput(uint i, float* output)
{
	float* inst = &(this->output[i * outVars]);
	memcpy(inst, output, outVars * sizeof(float));
}

//===========================================================================//

}
