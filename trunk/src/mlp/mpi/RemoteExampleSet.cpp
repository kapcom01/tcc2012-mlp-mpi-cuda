#include "mlp/mpi/RemoteExampleSet.h"

namespace ParallelMLP
{

//===========================================================================//

RemoteExampleSet::RemoteExampleSet(uint size, uint inVars, uint outVars)
	: ExampleSet(size, inVars, outVars)
{
	HostExampleSet set(size, inVars, outVars);
	copyToMaster(set);
}

//===========================================================================//

RemoteExampleSet::RemoteExampleSet(const Relation& relation)
	: ExampleSet(relation)
{
	HostExampleSet set(relation);
	copyToMaster(set);
}

//===========================================================================//

RemoteExampleSet::~RemoteExampleSet()
{

}

//===========================================================================//

void RemoteExampleSet::copyToMaster(const HostExampleSet &set)
{
	hid = COMM_WORLD.Get_rank();

	// Aloca os vetores
	input = new float[size * step];
	output = new float[size * outVars];
	stat = new Stat[step];

	memcpy(input, set.getInput(), size * step * sizeof(float));
	memcpy(stat, set.getStat(), step * sizeof(Stat));
}

//===========================================================================//

void RemoteExampleSet::normalize()
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

void RemoteExampleSet::unnormalize()
{
	if (!isNormalized)
		return;

	// Desnormaliza cada entrada
	for (uint i = 0; i < size * step; i++)
	{
		uint j = i % step;
		adjust(input[i], stat[j].to, stat[j].from);
	}

	// Desnormalização da saída é feita apenas no mestre
	if (hid == 0)
	{
		// Desnormaliza cada saída da rede neural
		for (uint i = 0; i < size * outVars; i++)
		{
			uint j = i % outVars + inVars;
			adjust(output[i], stat[j].to, stat[j].from);
		}
	}

	isNormalized = false;
}

//===========================================================================//

void RemoteExampleSet::adjust(float &x, const Range &from, const Range &to)
		const
{
	x = (to.upper - to.lower) / (from.upper - from.lower)
			* (x - from.lower) + to.lower;
}

//===========================================================================//

void RemoteExampleSet::setOutput(uint i, float* output)
{
	// Insere a saída apenas no mestre
	if (hid == 0)
	{
		float* inst = &(this->output[i * outVars]);
		memcpy(inst, output, outVars * sizeof(float));
	}
}

//===========================================================================//

}
