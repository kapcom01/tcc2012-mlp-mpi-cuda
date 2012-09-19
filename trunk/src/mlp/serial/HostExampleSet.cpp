#include "mlp/serial/HostExampleSet.h"

namespace ParallelMLP
{

void adjust(float* x, const Range* from, const Range* to);

//===========================================================================//

HostExampleSet::HostExampleSet(int relationID, int mlpID, SetType type)
	: ExampleSet(relationID, mlpID, type)
{

}

//===========================================================================//

HostExampleSet::~HostExampleSet()
{

}

//===========================================================================//

void HostExampleSet::normalize()
{
	if (isNormalized)
		return;

	Vector<float> input(this->input, inVars);
	Vector<float> target(this->target, outVars);
	Vector<Stat> inStat(this->inStat);
	Vector<Stat> outStat(this->outStat);

	// Para cada instância
	for (uint k = 0; k < size; k++)
	{
		// Normaliza cada coluna de entrada
		for (uint i = 0; i < inVars; i++)
			adjust(&(input[k][i]), &(inStat[i]->from), &(inStat[i]->to));

		// Normaliza cada coluna de saída
		for (uint t = 0; t < outVars; t++)
			adjust(&(target[k][t]), &(outStat[t]->from), &(outStat[t]->to));
	}

	isNormalized = true;
}

//===========================================================================//

void HostExampleSet::unnormalize()
{
	if (!isNormalized)
		return;

	Vector<float> input(this->input, inVars);
	Vector<float> target(this->target, outVars);
	Vector<float> output(this->output, outVars);
	Vector<Stat> inStat(this->inStat);
	Vector<Stat> outStat(this->outStat);

	// Para cada instância
	for (uint k = 0; k < size; k++)
	{
		// Para cada coluna de entrada
		for (uint i = 0; i < inVars; i++)
			adjust(&(input[k][i]), &(inStat[i]->to), &(inStat[i]->from));

		// Para cada coluna de saída
		for (uint t = 0; t < outVars; t++)
		{
			adjust(&(target[k][t]), &(outStat[t]->to), &(outStat[t]->from));
			adjust(&(output[k][t]), &(outStat[t]->to), &(outStat[t]->from));
		}
	}

	isNormalized = false;
}

//===========================================================================//

void adjust(float* x, const Range* from, const Range* to)
{
	*x = (to->upper - to->lower) / (from->upper - from->lower)
			* (*x - from->lower) + to->lower;
}

//===========================================================================//


}
