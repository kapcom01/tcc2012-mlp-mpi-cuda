#include "mlp/cuda/DeviceExampleSet.h"

namespace ParallelMLP
{

__host__ __device__
void adjust(float* x, const Range* from, const Range* to);

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

__global__
void normalizeVec(vec_float vec, vec_stat stat, uint offset)
{
	int k = blockIdx.x + offset;
	int i = threadIdx.x;

	adjust(&(vec(k)[i]), &(stat(i)->from), &(stat(i)->to));
}

//===========================================================================//

__global__
void unnormalizeVec(vec_float vec, vec_stat stat, uint offset)
{
	int k = blockIdx.x + offset;
	int i = threadIdx.x;

	adjust(&(vec(k)[i]), &(stat(i)->to), &(stat(i)->from));
}

//===========================================================================//

void DeviceExampleSet::copyToDevice()
{
	// Copia os dados da memória para a GPU
	devInput = input;
	devTarget = target;
	devOutput = output;
	devInStat = inStat;
	devOutStat = outStat;

	// Atribui os ponteiros puros
	rawInput = vec_float(devInput, inVars);
	rawTarget = vec_float(devTarget, outVars);
	rawOutput = vec_float(devOutput, outVars);
	rawInStat = vec_stat(devInStat);
	rawOutStat = vec_stat(devOutStat);
}

//===========================================================================//

void DeviceExampleSet::copyToHost()
{
	// Copia os dados da GPU para a memória
	input = devInput;
	target = devTarget;
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

		// Normaliza as colunas de entrada
		normalizeVec<<<blocks, inVars>>>(rawInput, rawInStat, i);

		// Normaliza as colunas de saída alvo
		normalizeVec<<<blocks, outVars>>>(rawTarget, rawOutStat, i);
	}

	isNormalized = true;
}

//===========================================================================//

void DeviceExampleSet::unnormalize()
{
	if (!isNormalized)
		return;

	for (uint i = 0; i < size; i += MAX_BLOCKS)
	{
		uint blocks = (size - i >= MAX_BLOCKS) ? MAX_BLOCKS : (size - i);

		// Normaliza as colunas de entrada
		unnormalizeVec<<<blocks, inVars>>>(rawInput, rawInStat, i);

		// Normaliza as colunas de saída alvo
		unnormalizeVec<<<blocks, outVars>>>(rawTarget, rawOutStat, i);

		// Normaliza as colunas de saída da rede neural
		unnormalizeVec<<<blocks, outVars>>>(rawOutput, rawOutStat, i);
	}

	// Copia os dados de volta para a memória
	copyToHost();

	isNormalized = false;
}

//===========================================================================//

vec_float DeviceExampleSet::getInput(uint i)
{
	if (isNormalized)
		return vec_float(devInput, inVars, i);
	else
		return ExampleSet::getInput(i);
}

//===========================================================================//

vec_float DeviceExampleSet::getTarget(uint i)
{
	if (isNormalized)
		return vec_float(devTarget, outVars, i);
	else
		return ExampleSet::getTarget(i);
}

//===========================================================================//

void DeviceExampleSet::setOutput(uint i, const vec_float output)
{
	vec_float this_out(this->devOutput, outVars, i);
	cudaMemcpy(this_out.data, output.data, outVars * sizeof(float),
			cudaMemcpyDeviceToDevice);
}

//===========================================================================//

__host__ __device__
void adjust(float* x, const Range* from, const Range* to)
{
	*x = (to->upper - to->lower) / (from->upper - from->lower)
			* (*x - from->lower) + to->lower;
}

//===========================================================================//

}
