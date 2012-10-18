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
	error.resize(outUnits);
	totalError.resize(1);

	rerror = error.data().get();
	rtotalError = error.data().get();
}

//===========================================================================//

DeviceOutLayer::~DeviceOutLayer()
{

}

//===========================================================================//

__global__
void calculateDiff(const float* target, float* output, uint outUnits,
		float* error, float* totalError)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < outUnits)
	{
		error[i] = target[i] - output[i];
		*totalError += error[i] * error[i];
		//atomicAdd(totalError, error[i] * error[i]);
	}
}

//===========================================================================//

void DeviceOutLayer::calculateError(const float* target)
{
	float aux = totalError[0];
	totalError[0] = 0;

	// Calcula a diferença da saída alvo com a saída gerada
	calculateDiff<<<outBlocks, TPB>>>(target, rfuncSignal, outUnits, rerror,
			rtotalError);

	totalError[0] = (aux * samples + totalError[0]) / (samples + outUnits);
	samples += outUnits;
}

//===========================================================================//

void DeviceOutLayer::feedback(const float* target, float learning)
{
	// Calcula o erro e chama o feedback do pai
	calculateError(target);
	DeviceLayer::feedback(rerror, learning);
}

//===========================================================================//

void DeviceOutLayer::clearTotalError()
{
	totalError[0] = 0;
	samples = 0;
}

//===========================================================================//

float DeviceOutLayer::getTotalError()
{
	return totalError[0];
}

//===========================================================================//



}

