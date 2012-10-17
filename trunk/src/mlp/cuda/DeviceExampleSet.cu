#include "mlp/cuda/DeviceExampleSet.h"

namespace ParallelMLP
{

__host__ __device__
void d_adjust(float &x, const Range &from, const Range &to);

//===========================================================================//

DeviceExampleSet::DeviceExampleSet(int relationID, int mlpID, SetType type)
	: ExampleSet(relationID, mlpID, type)
{

}

//===========================================================================//

DeviceExampleSet::~DeviceExampleSet()
{

}

//===========================================================================//

void DeviceExampleSet::copyToDevice()
{
	// Copia os dados da memória para a GPU
	devInput = input;
	devOutput = output;
	devStat = stat;

	// Atribui os ponteiros puros
	rawInput = vec_float(devInput, inVars + outVars);
	rawOutput = vec_float(devOutput, outVars);
	rawStat = vec_stat(devStat);
}

//===========================================================================//

void DeviceExampleSet::copyToHost()
{
	// Copia os dados da GPU para a memória
	input = devInput;
	output = devOutput;
}

//===========================================================================//

__global__
void normalizeVec(vec_float vec, vec_stat stat, uint offset)
{
	int k = blockIdx.x + offset;
	int i = threadIdx.x;

	d_adjust(vec(k)[i], stat(i)->from, stat(i)->to);
}

//===========================================================================//

void DeviceExampleSet::normalize()
{
	if (isNormalized)
		return;

	// Copia os dados para o dispositivo
	copyToDevice();

	for (uint i = 0; i < size; i += MAX_BLOCKS)
	{
		uint blocks = (size - i >= MAX_BLOCKS) ? MAX_BLOCKS : (size - i);

		// Normaliza as colunas de dados
		normalizeVec<<<blocks, inVars + outVars>>>(rawInput, rawStat, i);
	}

	copyToHost();

	isNormalized = true;
}

//===========================================================================//

__global__
void unnormalizeVec(vec_float vec, vec_stat stat, uint offset, uint statOffset)
{
	int k = blockIdx.x + offset;
	int i = threadIdx.x;

	d_adjust(vec(k)[i], stat(i + statOffset)->to, stat(i + statOffset)->from);
}

//===========================================================================//

void DeviceExampleSet::unnormalize()
{
	if (!isNormalized)
		return;

	for (uint i = 0; i < size; i += MAX_BLOCKS)
	{
		uint blocks = (size - i >= MAX_BLOCKS) ? MAX_BLOCKS : (size - i);

		// Normaliza as colunas de dados
		unnormalizeVec<<<blocks, inVars + outVars>>>(rawInput, rawStat, i, 0);

		// Normaliza as colunas de saída da rede neural
		unnormalizeVec<<<blocks, outVars>>>(rawOutput, rawStat, i, inVars);
	}

	// Copia os dados de volta para a memória
	copyToHost();

	isNormalized = false;
}

//===========================================================================//

vec_float DeviceExampleSet::getInput(uint i)
{
	if (isNormalized)
		return vec_float(devInput, inVars + outVars, i, inVars);
	else
		return ExampleSet::getInput(i);
}

//===========================================================================//

vec_float DeviceExampleSet::getTarget(uint i)
{
	if (isNormalized)
		return vec_float(devInput, inVars + outVars, i, outVars, inVars);
	else
		return ExampleSet::getTarget(i);
}

//===========================================================================//

void DeviceExampleSet::setOutput(uint i, vec_float &output)
{
	cout << "Hello from DeviceExampleSet::setOutput" << endl;
	vec_float this_out(this->devOutput, outVars, i, outVars);
	this_out.deviceCopyTo(output);
}

//===========================================================================//

__host__ __device__
void d_adjust(float &x, const Range &from, const Range &to)
{
	x = (to.upper - to.lower) / (from.upper - from.lower)
			* (x - from.lower) + to.lower;
}

//===========================================================================//

}
