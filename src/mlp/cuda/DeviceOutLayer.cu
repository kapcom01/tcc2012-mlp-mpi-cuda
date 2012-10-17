#include "mlp/cuda/DeviceOutLayer.h"

namespace ParallelMLP
{

//===========================================================================//

DeviceOutLayer::DeviceOutLayer()
{

}

//===========================================================================//

DeviceOutLayer::DeviceOutLayer(uint inUnits, uint outUnits)
{
	init(inUnits, outUnits);
}

//===========================================================================//

void DeviceOutLayer::init(uint inUnits, uint outUnits)
{
	DeviceLayer::init(inUnits, outUnits);

	error.resize(outUnits);
	error2.resize(outUnits);

	rawError = vec_float(error);
	rawError2 = vec_float(error2);
}

//===========================================================================//

DeviceOutLayer::~DeviceOutLayer()
{

}

//===========================================================================//

__global__
void calculateDiff(vec_float error, vec_float error2, vec_float target,
		vec_float output)
{
	int n = blockIdx.x;

	error[n] = target[n] - output[n];
	error2[n] = error[n] * error[n];
}

//===========================================================================//

void DeviceOutLayer::calculateError(const vec_float &target)
{
	// Calcula a diferença da saída alvo com a saída gerada
	calculateDiff<<<outUnits, 1>>>(rawError, rawError2, target, rawFuncSignal);

	// Calcula o erro total
	float inc = thrust::reduce(error2.begin(), error2.end());

	cout << "       |-> Target: ";
	vec_float aux = target;
	device_ptr<float> ptr = thrust::device_pointer_cast(aux.data());
	dv_float sig(ptr, ptr + inUnits);
	for (uint i = 0; i < outUnits; i++)
		cout << sig[i] << " ";
	cout << endl;

	// Incrementa o erro
	incTotalError(inc, outUnits);
}

//===========================================================================//

void DeviceOutLayer::feedback(const vec_float &target, float learning)
{
	// Calcula o erro e chama o feedback do pai
	calculateError(target);
	DeviceLayer::feedback(rawError, learning);
}

//===========================================================================//

}

