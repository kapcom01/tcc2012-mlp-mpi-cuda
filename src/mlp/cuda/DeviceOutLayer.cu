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
	rawError = vec_float(error);
}

//===========================================================================//

DeviceOutLayer::~DeviceOutLayer()
{

}

//===========================================================================//

__global__
void calculateDiff(vec_float error, vec_float target, vec_float output)
{
	int n = blockIdx.x;

	error[n] = target[n] - output[n];
}

//===========================================================================//

struct square : public thrust::unary_function<float, float>
{
	__host__ __device__
	float operator()(float x) const
	{
		return x * x;
	}
};

//===========================================================================//

void DeviceOutLayer::calculateError(const vec_float &target)
{
	// Calcula a diferença da saída alvo com a saída gerada
	calculateDiff<<<outUnits, 1>>>(rawError, target, rawFuncSignal);

	// Calcula o erro total
	float inc = thrust::reduce(
			thrust::make_transform_iterator(error.begin(), square()),
			thrust::make_transform_iterator(error.end(), square()));

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

