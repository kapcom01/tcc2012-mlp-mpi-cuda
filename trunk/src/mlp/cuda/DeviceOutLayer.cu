#include "mlp/cuda/DeviceOutLayer.h"

namespace ParallelMLP
{

//===========================================================================//

DeviceOutLayer::DeviceOutLayer(uint inUnits, uint outUnits)
	: DeviceLayer(inUnits, outUnits)
{
	init(inUnits, outUnits);
}

//===========================================================================//

void DeviceOutLayer::init(uint inUnits, uint outUnits)
{
	cudaMalloc(&error, outUnits * sizeof(float));
	cudaMalloc(&sum, sizeof(float));
}

//===========================================================================//

DeviceOutLayer::~DeviceOutLayer()
{

}

//===========================================================================//

__global__
void calculateDiff(const float* target, float* output, uint outUnits,
		float* error, float* sum)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < outUnits)
	{
		error[i] = target[i] - output[i];
		*sum += error[i] * error[i];
		//atomicAdd(totalError, error[i] * error[i]);
	}
}

//===========================================================================//

void DeviceOutLayer::calculateError(const float* target)
{
	cudaMemset(sum, 0, sizeof(float));

	// Calcula a diferença da saída alvo com a saída gerada
	calculateDiff<<<outBlocks, TPB>>>(target, funcSignal, outUnits, error,
			sum);

	float aux;
	cudaMemcpy(&aux, &sum, sizeof(float), cudaMemcpyDeviceToHost);

	totalError = (totalError * samples + aux) / (samples + outUnits);
	samples += outUnits;
}

//===========================================================================//

void DeviceOutLayer::feedback(const float* target, float learning)
{
	// Calcula o erro e chama o feedback do pai
	calculateError(target);
	DeviceLayer::feedback(error, learning);
}

//===========================================================================//

void DeviceOutLayer::clearTotalError()
{
	totalError = 0;
	samples = 0;
}

//===========================================================================//

float DeviceOutLayer::getTotalError()
{
	return totalError;
}

//===========================================================================//



}

