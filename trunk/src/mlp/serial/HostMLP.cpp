#include "mlp/serial/HostMLP.h"

namespace ParallelMLP
{

//===========================================================================//

HostMLP::HostMLP(int mlpID)
	: MLP(mlpID)
{

}

//===========================================================================//

HostMLP::HostMLP(string name, v_uint &units)
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

HostMLP::~HostMLP()
{

}

//===========================================================================//

void HostMLP::addLayer(uint inUnits, uint outUnits, bool isOutput)
{
	if (isOutput)
		layers.push_back(new HostOutLayer(inUnits, outUnits));
	else
		layers.push_back(new HostLayer(inUnits, outUnits));
}

//===========================================================================//

void HostMLP::train(HostExampleSet* training)
{
	MLP::train(training);
}

//===========================================================================//

void HostMLP::validate(HostExampleSet* validation)
{
	MLP::validate(validation);
}

//===========================================================================//

void HostMLP::test(HostExampleSet* test)
{
	MLP::test(test);
}

//===========================================================================//

}
