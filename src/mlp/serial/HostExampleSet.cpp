#include "mlp/serial/HostExampleSet.h"

namespace ParallelMLP
{

void adjust(float &x, const Range &from, const Range &to);

//===========================================================================//

HostExampleSet::HostExampleSet()
{

}

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

	Vector<float> input(this->input, inVars + outVars);
	Vector<Stat> stat(this->stat);

	// Para cada instância
	for (uint k = 0; k < size; k++)
	{
		// Normaliza cada coluna de dados
		for (uint i = 0; i < inVars + outVars; i++)
			adjust(input(k)[i], stat(i)->from, stat(i)->to);
	}

	isNormalized = true;
}

//===========================================================================//

void HostExampleSet::unnormalize()
{
	if (!isNormalized)
		return;

	Vector<float> input(this->input, inVars + outVars);
	Vector<float> output(this->output, outVars);
	Vector<Stat> stat(this->stat);

	// Para cada instância
	for (uint k = 0; k < size; k++)
	{
		// Para cada coluna de dados
		for (uint i = 0; i < inVars + outVars; i++)
			adjust(input(k)[i], stat(i)->to, stat(i)->from);

		// Para cada coluna de saída da rede neural
		for (uint t = 0; t < outVars; t++)
			adjust(output(k)[t], stat(t + inVars)->to, stat(t + inVars)->from);
	}

	isNormalized = false;
}

//===========================================================================//

void adjust(float &x, const Range &from, const Range &to)
{
	x = (to.upper - to.lower) / (from.upper - from.lower)
			* (x - from.lower) + to.lower;
}

//===========================================================================//


}
