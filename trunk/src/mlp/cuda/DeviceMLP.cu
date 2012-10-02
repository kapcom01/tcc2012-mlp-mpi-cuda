#include "mlp/cuda/DeviceMLP.h"

namespace ParallelMLP
{

//===========================================================================//

DeviceMLP::DeviceMLP(int mlpID)
	: MLP(mlpID)
{
	srand(time(NULL));
}

//===========================================================================//

DeviceMLP::DeviceMLP(string name, vector<uint> &units)
	: MLP(name, units)
{
	// Adiciona as camadas escondidas e a camada de saída
	for (uint i = 0; i < units.size() - 1; i++)
		addLayer(units[i], units[i + 1]);

	// Seta a saída e randomiza os pesos
	setOutput();
	randomize();
}

//===========================================================================//

DeviceMLP::~DeviceMLP()
{

}

//===========================================================================//

void DeviceMLP::addLayer(uint inUnits, uint outUnits)
{
	layers.push_back(new DeviceLayer(inUnits, outUnits));
}

//===========================================================================//

void DeviceMLP::train(DeviceExampleSet &training)
{
	MLP::train(training);
}

//===========================================================================//

void DeviceMLP::validate(DeviceExampleSet &validation)
{
	MLP::validate(validation);
}

//===========================================================================//

void DeviceMLP::test(DeviceExampleSet &test)
{
	MLP::test(test);
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

void DeviceMLP::calculateError(const vec_float target)
{
	// Calcula a diferença da saída alvo com a saída gerada
	calculateDiff<<<error.size(), 1>>>(rawError, target, output);

	// Calcula o erro total
	float totalError = thrust::reduce(
			thrust::make_transform_iterator(error.begin(), square()),
			thrust::make_transform_iterator(error.end(), square()));
}

//===========================================================================//

void DeviceMLP::setOutput()
{
	MLP::setOutput();

	error.resize(layers.back()->getOutUnits());
	rawError = vec_float(error);
}

//===========================================================================//

}
