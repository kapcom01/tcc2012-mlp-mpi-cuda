#include "mlp/cuda/DeviceOutLayer.h"

namespace ParallelMLP
{

//===========================================================================//

DeviceOutLayer::DeviceOutLayer(uint inUnits, uint outUnits)
	: DeviceLayer(inUnits, outUnits)
{
	cudaMalloc(&error, outUnits * sizeof(float));
	cudaMalloc(&sum, sizeof(float));
	totalError = 0;
	samples = 0;
}

//===========================================================================//

DeviceOutLayer::~DeviceOutLayer()
{
	cudaFree(error);
	cudaFree(sum);
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

	// Recupera a soma dos erros
	float hsum;
	cudaMemcpy(&hsum, &sum, sizeof(float), cudaMemcpyDeviceToHost);

	// Calcula o erro quadrático médio
	totalError = (totalError * samples + hsum) / (samples + outUnits);
	samples += outUnits;
}

//===========================================================================//

void DeviceOutLayer::feedbackward(const float* target, float learning)
{
	// Calcula o erro e chama o feedback do pai
	calculateError(target);
	DeviceLayer::feedbackward(error, learning);
}

//===========================================================================//

void DeviceOutLayer::clearError()
{
	totalError = 0;
	samples = 0;
}

//===========================================================================//

float DeviceOutLayer::getError()
{
	return totalError;
}

//===========================================================================//



}

