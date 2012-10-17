#include "mlp/cuda/DeviceMLP.h"

namespace ParallelMLP
{

//===========================================================================//

DeviceMLP::DeviceMLP(int mlpID)
	: MLP(mlpID)
{

}

//===========================================================================//

DeviceMLP::DeviceMLP(string name, v_uint &units)
	: MLP(name, units)
{
	// Adiciona as camadas escondidas e a camada de saída
	for (uint i = 0; i < units.size() - 1; i++)
		addLayer(units[i], units[i + 1], i == units.size() - 2);

	// Seta a saída e randomiza os pesos
	config();
	randomize();
}

//===========================================================================//

DeviceMLP::~DeviceMLP()
{

}

//===========================================================================//

void DeviceMLP::addLayer(uint inUnits, uint outUnits, bool isOutput)
{
	if (isOutput)
		layers.push_back(new DeviceOutLayer(inUnits, outUnits));
	else
		layers.push_back(new DeviceLayer(inUnits, outUnits));
}

//===========================================================================//

void DeviceMLP::train(DeviceExampleSet* training)
{
	MLP::train(training);
}

//===========================================================================//

void DeviceMLP::validate(DeviceExampleSet* validation)
{
	MLP::validate(validation);
}

//===========================================================================//

void DeviceMLP::test(DeviceExampleSet* test)
{
	MLP::test(test);
}

//===========================================================================//

}
