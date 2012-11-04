#include "mlp/cuda/DeviceOutLayer.h"

namespace ParallelMLP
{

//===========================================================================//

DeviceOutLayer::DeviceOutLayer(uint inUnits, uint outUnits)
	: Layer(inUnits, outUnits), OutLayer(inUnits, outUnits),
	  DeviceLayer(inUnits, outUnits)
{
	cudaMalloc(&error, outUnits * sizeof(float));
	cudaMalloc(&sum, sizeof(float));
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

	// Incrementa o erro
	incError(hsum);
}

//===========================================================================//

void DeviceOutLayer::feedbackward(const float* target, float learning)
{
	// Calcula o erro e chama o feedback do pai
	calculateError(target);
	DeviceLayer::feedbackward(error, learning);
}

//===========================================================================//

}

