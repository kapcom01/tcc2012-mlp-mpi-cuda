#include "mlp/serial/HostMLP.h"

namespace ParallelMLP
{

//===========================================================================//

HostMLP::HostMLP(int mlpID)
	: MLP(mlpID)
{
	srand(time(NULL));
}

//===========================================================================//

HostMLP::HostMLP(string name, vector<uint> &units)
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

HostMLP::~HostMLP()
{

}

//===========================================================================//

void HostMLP::addLayer(uint inUnits, uint outUnits)
{
	layers.push_back(new HostLayer(inUnits, outUnits));
}

//===========================================================================//

void HostMLP::train(HostExampleSet &training)
{
	MLP::train(training);
}

//===========================================================================//

void HostMLP::validate(HostExampleSet &validation)
{
	MLP::validate(validation);
}

//===========================================================================//

void HostMLP::test(HostExampleSet &test)
{
	MLP::test(test);
}

//===========================================================================//

void HostMLP::calculateError(const vec_float target)
{
	for (uint i = 0; i < error.size(); i++)
	{
		error[i] = target[i] - output[i];
		totalError += error[i] * error[i];
	}
}

//===========================================================================//

void HostMLP::setOutput()
{
	MLP::setOutput();

	error.resize(layers.back()->getOutUnits());
	rawError = vec_float(error);
}

//===========================================================================//

}
