#include "mlp/mpi/RemoteMLP.h"

namespace ParallelMLP
{

//===========================================================================//

RemoteMLP::RemoteMLP(int mlpID)
	: MLP(mlpID)
{
	hid = COMM_WORLD.Get_rank();
	hosts = COMM_WORLD.Get_size();
}

//===========================================================================//

RemoteMLP::RemoteMLP(string name, v_uint &units)
	: MLP(name, units)
{
	hid = COMM_WORLD.Get_rank();
	hosts = COMM_WORLD.Get_size();

	// Adiciona as camadas escondidas e a camada de saída
	for (uint i = 0; i < units.size() - 1; i++)
		addLayer(units[i], units[i + 1], i == units.size() - 2);

	// Seta a saída e randomiza os pesos
	config();
	randomize();
}

//===========================================================================//

RemoteMLP::~RemoteMLP()
{

}

//===========================================================================//

void RemoteMLP::addLayer(uint inUnits, uint outUnits, bool isOutput)
{
	if (isOutput)
		layers.push_back(new RemoteOutLayer(inUnits, outUnits, hid, hosts));
	else
		layers.push_back(new RemoteLayer(inUnits, outUnits, hid, hosts));
}

//===========================================================================//

void RemoteMLP::train(RemoteExampleSet* training)
{
	MLP::train(training);
}

//===========================================================================//

void RemoteMLP::validate(RemoteExampleSet* validation)
{
	MLP::validate(validation);
}

//===========================================================================//

void RemoteMLP::test(RemoteExampleSet* test)
{
	MLP::test(test);
}

//===========================================================================//

}
