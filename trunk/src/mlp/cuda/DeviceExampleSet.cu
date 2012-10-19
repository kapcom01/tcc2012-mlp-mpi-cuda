#include "mlp/cuda/DeviceExampleSet.h"

namespace ParallelMLP
{

__device__
void adjust(float &x, const Range &from, const Range &to);

//===========================================================================//

DeviceExampleSet::DeviceExampleSet(const Relation& relation)
{
	HostExampleSet set(relation);

	// Recupera os tamanhos
	size = set.getSize();
	inVars = set.getInVars();
	outVars = set.getOutVars();
	step = inVars + outVars;
	stepBlocks = (size * step) / TPB + 1;
	outBlocks = (size * outVars) / TPB + 1;

	// Aloca espaço no dispositivo
	cudaMalloc(&input, size * step * sizeof(float));
	cudaMalloc(&output, size * outVars * sizeof(float));
	cudaMalloc(&stat, step * sizeof(Stat));

	copyToDevice(set);
}

//===========================================================================//

DeviceExampleSet::~DeviceExampleSet()
{
	cudaFree(input);
	cudaFree(output);
	cudaFree(stat);
}

//===========================================================================//

void DeviceExampleSet::copyToDevice(const HostExampleSet &set)
{
	// Copia os dados para o dispositivo
	cudaMemcpy(input, set.getInput(), size * step * sizeof(float),
			cudaMemcpyHostToDevice);
	cudaMemcpy(stat, set.getStat(), step * sizeof(Stat),
			cudaMemcpyHostToDevice);
}

//===========================================================================//

__global__
void normalizeVec(float* vec, Stat* stat, uint size, uint step, uint offset)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = i % step + offset;

	if (i < size * step)
		adjust(vec[i], stat[j].from, stat[j].to);
}

//===========================================================================//

void DeviceExampleSet::normalize()
{
	if (isNormalized)
		return;

	// Normaliza as entradas
	normalizeVec<<<stepBlocks, TPB>>>(input, stat, size, step, 0);

	isNormalized = true;
}

//===========================================================================//

__global__
void unnormalizeVec(float* vec, Stat* stat, uint size, uint step, uint offset)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = i % step + offset;

	if (i < size * step)
		adjust(vec[i], stat[j].to, stat[j].from);
}

//===========================================================================//

void DeviceExampleSet::unnormalize()
{
	if (!isNormalized)
		return;

	// Desnormaliza as entradas
	unnormalizeVec<<<stepBlocks, TPB>>>(input, stat, size, step, 0);

	// Desnormaliza as saídas
	unnormalizeVec<<<outBlocks, TPB>>>(output, stat, size, outVars, inVars);

	isNormalized = false;
}

//===========================================================================//

__device__
void adjust(float &x, const Range &from, const Range &to)
{
	x = (to.upper - to.lower) / (from.upper - from.lower)
			* (x - from.lower) + to.lower;
}

//===========================================================================//

uint DeviceExampleSet::getInVars() const
{
	return inVars;
}

//===========================================================================//

uint DeviceExampleSet::getOutVars() const
{
	return outVars;
}

//===========================================================================//

uint DeviceExampleSet::getSize() const
{
	return size;
}

//===========================================================================//

const float* DeviceExampleSet::getInput(uint i) const
{
	return &input[i * step];
}

//===========================================================================//

const float* DeviceExampleSet::getTarget(uint i) const
{
	return &input[i * step + inVars];
}

//===========================================================================//

void DeviceExampleSet::setOutput(uint i, float* output)
{
	float* inst = &(this->output[i * outVars]);
	cudaMemcpy(inst, output, outVars * sizeof(float),
			cudaMemcpyDeviceToDevice);
}

//===========================================================================//

float DeviceExampleSet::getLearning() const
{
	return learning;
}

//===========================================================================//

void DeviceExampleSet::setLearning(float learning)
{
	this->learning = learning;
}

//===========================================================================//

float DeviceExampleSet::getTolerance() const
{
	return tolerance;
}

//===========================================================================//

void DeviceExampleSet::setTolerance(float tolerance)
{
	this->tolerance = tolerance;
}

//===========================================================================//

uint DeviceExampleSet::getMaxEpochs() const
{
	return maxEpochs;
}

//===========================================================================//

void DeviceExampleSet::setMaxEpochs(uint maxEpochs)
{
	this->maxEpochs = maxEpochs;
}

//===========================================================================//

float DeviceExampleSet::getError() const
{
	return error;
}

//===========================================================================//

void DeviceExampleSet::setError(float error)
{
	this->error = error;
}

//===========================================================================//

uint DeviceExampleSet::getEpochs() const
{
	return epochs;
}

//===========================================================================//

void DeviceExampleSet::setEpochs(uint epochs)
{
	this->epochs = epochs;
}

//===========================================================================//

float DeviceExampleSet::getTime() const
{
	return time;
}

//===========================================================================//

void DeviceExampleSet::setTime(float time)
{
	this->time = time;
}

//===========================================================================//

}
